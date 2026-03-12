use core::fmt;
use std::error::Error;

use indexmap::IndexMap;
use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct ValveKvParser;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Block<'a> {
    pub properties: IndexMap<&'a str, &'a str>,
}

#[derive(Debug)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parse Error: {}", self.0)
    }
}

impl Error for ParseError {}

impl<'a> Block<'a> {
    fn parse_from_pair(pair: pest::iterators::Pair<'a, Rule>) -> Result<Self, ParseError> {
        let mut properties = IndexMap::new();
        for prop_pair in pair.into_inner() {
            if prop_pair.as_rule() == Rule::property {
                let (key, value) = Self::parse_property(prop_pair)?;
                properties.insert(key, value);
            }
        }
        Ok(Self { properties })
    }

    fn parse_property(
        pair: pest::iterators::Pair<'a, Rule>,
    ) -> Result<(&'a str, &'a str), ParseError> {
        let mut inner = pair.into_inner();
        let key = inner
            .next()
            .ok_or_else(|| ParseError("Missing key in property".into()))?;
        let value = inner
            .next()
            .ok_or_else(|| ParseError("Missing value in property".into()))?;

        Ok((Self::extract_string(key), Self::extract_string(value)))
    }

    fn extract_string(p: pest::iterators::Pair<'a, Rule>) -> &'a str {
        let s = p.as_str();
        s.strip_prefix('"')
            .and_then(|str| str.strip_suffix('"'))
            .unwrap_or(s)
    }
}

pub fn parse<'a>(input: &'a str) -> Result<Vec<Block<'a>>, ParseError> {
    let blocks_pairs =
        ValveKvParser::parse(Rule::blocks, input).map_err(|e| ParseError(e.to_string()))?;

    let mut blocks = Vec::new();
    for pair in blocks_pairs {
        if pair.as_rule() == Rule::block {
            blocks.push(Block::parse_from_pair(pair)?);
        }
    }

    Ok(blocks)
}
