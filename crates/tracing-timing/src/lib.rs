//! TreeTimingLayer that renders spans and events as a timing tree.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::io::Write as IoWrite;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use tracing::Level;
use tracing::field::Visit;

const DIM: &str = "\x1b[90m";
const RESET: &str = "\x1b[0m";

pub struct AlternateScreenGuard {
    cb: Option<Box<dyn FnOnce()>>,
}

impl AlternateScreenGuard {
    pub fn new(cb: impl FnOnce() + 'static) -> Self {
        print!("\x1b[?1049h");
        std::io::stdout().flush().ok();
        Self {
            cb: Some(Box::new(cb)),
        }
    }
}

impl Drop for AlternateScreenGuard {
    fn drop(&mut self) {
        print!("\x1b[?1049l");
        std::io::stdout().flush().ok();
        if let Some(cb) = self.cb.take() {
            cb();
        }
    }
}

fn format_human_duration(d: Duration) -> String {
    let us = d.as_micros();
    if us < 1_000 {
        return format!("{us}µs");
    }
    if us < 1_000_000 {
        return format!("{:.2}ms", us as f64 / 1_000.0);
    }

    let secs = d.as_secs();
    if secs < 60 {
        return format!("{:.2}s", d.as_secs_f64());
    }
    if secs < 3_600 {
        let mins = secs / 60;
        let rem_secs = secs % 60;
        return format!("{mins}m {rem_secs}s");
    }

    let hours = secs / 3_600;
    let rem_mins = (secs % 3_600) / 60;
    format!("{hours}h {rem_mins}m")
}

fn level_prefix(level: &Level) -> &'static str {
    match *level {
        Level::ERROR => "\x1b[31mERROR\x1b[0m",
        Level::WARN => "\x1b[33mWARN\x1b[0m",
        Level::INFO => "\x1b[32mINFO\x1b[0m",
        Level::DEBUG => "\x1b[34mDEBUG\x1b[0m",
        Level::TRACE => "\x1b[35mTRACE\x1b[0m",
    }
}

#[derive(Clone, Default)]
pub struct TreeTimingLayer {
    inner: Arc<RwLock<TreeState>>,
}

#[derive(Clone, Eq, PartialEq, Hash)]
enum NodeKey {
    Span(tracing::span::Id),
    Event(u64),
}

#[derive(Default)]
struct TreeState {
    nodes: HashMap<NodeKey, Node>,
    next_event_id: u64,
}

struct EventNode {
    name: String,
    parent: Option<NodeKey>,
    level: Level,
    target: String,
}

struct SpanNode {
    name: String,
    parent: Option<NodeKey>,
    children: Vec<NodeKey>,
    start: Option<Instant>,
    elapsed: Option<Duration>,
    fields: Option<String>,
}

enum Node {
    Span(SpanNode),
    Event(EventNode),
}

impl Node {
    fn parent(&self) -> Option<&NodeKey> {
        match self {
            Node::Span(s) => s.parent.as_ref(),
            Node::Event(e) => e.parent.as_ref(),
        }
    }
}

impl<S> tracing_subscriber::Layer<S> for TreeTimingLayer
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let parent = ctx.current_span().id().map(|id| NodeKey::Span(id.clone()));
        let mut v = FieldVisitor::default();
        attrs.record(&mut v);

        let node = Node::Span(SpanNode {
            name: attrs.metadata().name().to_string(),
            parent: parent.clone(),
            children: Vec::new(),
            start: None,
            elapsed: v.elapsed_ns.map(Duration::from_nanos),
            fields: v.finish_fields(),
        });

        let key = NodeKey::Span(id.clone());
        let mut inner = self.inner.write().unwrap();
        inner.nodes.insert(key.clone(), node);

        if let Some(NodeKey::Span(pid)) = parent
            && let Some(Node::Span(p)) = inner.nodes.get_mut(&NodeKey::Span(pid))
        {
            p.children.push(key);
        }
    }

    fn on_record(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        _: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut inner = self.inner.write().unwrap();
        let Some(Node::Span(node)) = inner.nodes.get_mut(&NodeKey::Span(id.clone())) else {
            return;
        };

        let mut v = FieldVisitor::default();
        values.record(&mut v);

        if let Some(ns) = v.elapsed_ns {
            node.elapsed = Some(Duration::from_nanos(ns));
        }
        if let Some(fields) = v.finish_fields() {
            node.fields = Some(fields);
        }
    }

    fn on_enter(&self, id: &tracing::span::Id, _: tracing_subscriber::layer::Context<'_, S>) {
        {
            let mut inner = self.inner.write().unwrap();
            if let Some(Node::Span(node)) = inner.nodes.get_mut(&NodeKey::Span(id.clone())) {
                node.start.get_or_insert(Instant::now());
            }
        }
        self.print_tree(true);
    }

    fn on_exit(&self, id: &tracing::span::Id, _: tracing_subscriber::layer::Context<'_, S>) {
        {
            let mut inner = self.inner.write().unwrap();
            if let Some(Node::Span(node)) = inner.nodes.get_mut(&NodeKey::Span(id.clone())) {
                node.elapsed = Some(node.start.unwrap_or_else(Instant::now).elapsed());
            }
        }
        self.print_tree(true);
    }

    fn on_event(&self, event: &tracing::Event<'_>, ctx: tracing_subscriber::layer::Context<'_, S>) {
        let parent = event
            .parent()
            .cloned()
            .or_else(|| ctx.current_span().id().cloned())
            .map(NodeKey::Span);
        let mut v = FieldVisitor::default();
        event.record(&mut v);
        let name = v
            .message
            .unwrap_or_else(|| event.metadata().target().to_string());

        {
            let mut inner = self.inner.write().unwrap();
            let key = NodeKey::Event(inner.next_event_id);
            inner.next_event_id += 1;

            let target = v
                .log_target
                .unwrap_or_else(|| event.metadata().target().to_string());
            inner.nodes.insert(
                key.clone(),
                Node::Event(EventNode {
                    name,
                    parent: parent.clone(),
                    level: *event.metadata().level(),
                    target,
                }),
            );

            if let Some(NodeKey::Span(pid)) = parent
                && let Some(Node::Span(p)) = inner.nodes.get_mut(&NodeKey::Span(pid))
            {
                p.children.push(key);
            }
        }
        self.print_tree(true);
    }
}

impl TreeTimingLayer {
    pub fn print_tree(&self, clear: bool) {
        let inner = self.inner.read().unwrap();
        let roots: Vec<_> = inner
            .nodes
            .iter()
            .filter(|(_, n)| n.parent().is_none())
            .map(|(id, _)| id.clone())
            .collect();

        let mut lines = Vec::new();
        for (i, root) in roots.iter().enumerate() {
            Self::collect_lines(&inner, root, "", i + 1 == roots.len(), true, &mut lines);
        }

        if clear {
            print!("\x1b[2J\x1b[H");
        }
        for line in &lines {
            println!("{line}");
        }
        std::io::stdout().flush().ok();
    }

    fn collect_lines(
        inner: &TreeState,
        id: &NodeKey,
        prefix: &str,
        is_last: bool,
        is_root: bool,
        out: &mut Vec<String>,
    ) {
        let Some(node) = inner.nodes.get(id) else {
            return;
        };

        match node {
            Node::Span(span) => {
                let (branch, next_pad) = if is_root {
                    ("", "")
                } else if is_last {
                    ("└── ", "    ")
                } else {
                    ("├── ", "│   ")
                };

                let elapsed = span
                    .elapsed
                    .map(|e| format!("[ {} ]", format_human_duration(e)))
                    .unwrap_or_else(|| "...".to_string());

                let fields = span
                    .fields
                    .as_ref()
                    .map(|f| format!(" {{ {f} }}"))
                    .unwrap_or_default();

                out.push(format!(
                    "{DIM}SPAN{RESET}\t{DIM}{prefix}{branch}{RESET}{} {elapsed}{fields}",
                    span.name
                ));

                let next_prefix = if is_root {
                    prefix.to_string()
                } else {
                    format!("{prefix}{next_pad}")
                };
                for (i, child) in span.children.iter().enumerate() {
                    Self::collect_lines(
                        inner,
                        child,
                        &next_prefix,
                        i + 1 == span.children.len(),
                        false,
                        out,
                    );
                }
            }
            Node::Event(ev) => {
                let branch = if is_root {
                    ""
                } else if is_last {
                    "└> "
                } else {
                    "├> "
                };
                out.push(format!(
                    "{}\t{DIM}{prefix}{branch}{RESET}{DIM}@{}{RESET} {}",
                    level_prefix(&ev.level),
                    ev.target,
                    ev.name.replace('\n', " ")
                ));
            }
        }
    }
}

#[derive(Default)]
struct FieldVisitor {
    buf: String,
    message: Option<String>,
    elapsed_ns: Option<u64>,
    log_target: Option<String>,
}

impl FieldVisitor {
    fn push_field(&mut self, name: &str, v: impl std::fmt::Display) {
        if !self.buf.is_empty() {
            self.buf.push_str(", ");
        }
        write!(self.buf, "{name} = {v}").ok();
    }

    fn finish_fields(self) -> Option<String> {
        (!self.buf.is_empty()).then_some(self.buf)
    }
}

impl Visit for FieldVisitor {
    fn record_u64(&mut self, f: &tracing::field::Field, v: u64) {
        if f.name() == "elapsed_ns" {
            self.elapsed_ns = Some(v);
        } else {
            self.push_field(f.name(), v);
        }
    }

    fn record_f64(&mut self, f: &tracing::field::Field, v: f64) {
        if f.name() == "elapsed_ns" {
            self.elapsed_ns = Some(v as u64);
        } else {
            self.push_field(f.name(), v);
        }
    }

    fn record_str(&mut self, f: &tracing::field::Field, v: &str) {
        match f.name() {
            "log.target" => self.log_target = Some(v.to_string()),
            "message" | "error" => self.message = Some(v.to_string()),
            _ => self.push_field(f.name(), v),
        }
    }

    fn record_debug(&mut self, f: &tracing::field::Field, v: &dyn std::fmt::Debug) {
        let formatted = format!("{v:?}").trim_matches('"').to_string();
        match f.name() {
            "message" | "error" => self.message = Some(formatted),
            "log.target" => self.log_target = Some(formatted),
            _ => {
                if !self.buf.is_empty() {
                    self.buf.push_str(", ");
                }
                write!(self.buf, "{} = {:?}", f.name(), v).ok();
            }
        }
    }
}
