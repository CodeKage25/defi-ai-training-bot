use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::str::FromStr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeProposal {
    pub id: String,
    pub token_in: String,
    pub token_out: String,
    pub amount_in: f64,
    pub amount_usd: f64,
    pub chain: String,
    pub dex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub status: String, // "executed" | "rejected"
    pub reason: Option<String>,
    pub tx_hash: Option<String>,
}

pub struct ExecutionEngine {
    whitelisted_tokens: HashMap<String, HashSet<String>>,
    whitelisted_dexs: HashSet<String>,
    max_drawdown_percent: f64,
    max_position_size_usd: f64,
    current_portfolio_value: f64,
}

impl ExecutionEngine {
    pub fn new() -> Self {
        let mut whitelisted_tokens = HashMap::new();
        
        let eth_tokens: HashSet<String> = ["ETH", "WETH", "USDC", "USDT", "DAI", "WBTC"]
            .iter().map(|s| s.to_string()).collect();
        whitelisted_tokens.insert("ethereum".to_string(), eth_tokens);
        
        // Add other chains...
        let polygon_tokens: HashSet<String> = ["MATIC", "WMATIC", "USDC", "USDT", "DAI", "WETH"]
            .iter().map(|s| s.to_string()).collect();
        whitelisted_tokens.insert("polygon".to_string(), polygon_tokens);

        let mut whitelisted_dexs = HashSet::new();
        for dex in ["uniswap_v3", "sushiswap", "curve", "pancakeswap", "quickswap", "1inch"] {
            whitelisted_dexs.insert(dex.to_string());
        }

        Self {
            whitelisted_tokens,
            whitelisted_dexs,
            max_drawdown_percent: 3.0,
            max_position_size_usd: 10000.0,
            current_portfolio_value: 100000.0, // Mock
        }
    }

    pub fn validate_and_execute(&self, proposal: TradeProposal) -> ExecutionResult {
        println!("Validating proposal: {:?}", proposal);

        // 1. Whitelist Check
        if !self.check_whitelist(&proposal) {
            return ExecutionResult {
                status: "rejected".to_string(),
                reason: Some("Failed Whitelist Check".to_string()),
                tx_hash: None,
            };
        }

        // 2. Drawdown Check
        if !self.check_drawdown(&proposal) {
            return ExecutionResult {
                status: "rejected".to_string(),
                reason: Some("Failed Drawdown Check".to_string()),
                tx_hash: None,
            };
        }

        // 3. Solvent Check (Mock)
        if !self.check_solvent(&proposal) {
            return ExecutionResult {
                status: "rejected".to_string(),
                reason: Some("Failed Solvent Check".to_string()),
                tx_hash: None,
            };
        }

        // Execute
        ExecutionResult {
            status: "executed".to_string(),
            reason: None,
            tx_hash: Some(format!("0xrust{}", "1".repeat(60))),
        }
    }

    fn check_whitelist(&self, proposal: &TradeProposal) -> bool {
        // Check Chain
        let tokens = match self.whitelisted_tokens.get(&proposal.chain) {
            Some(t) => t,
            None => return false,
        };

        // Check Tokens
        if !tokens.contains(&proposal.token_in) || !tokens.contains(&proposal.token_out) {
            return false;
        }

        // Check DEX (partial match)
        if !self.whitelisted_dexs.iter().any(|d| proposal.dex.contains(d)) {
            return false;
        }

        true
    }

    fn check_drawdown(&self, proposal: &TradeProposal) -> bool {
        if proposal.amount_usd > self.max_position_size_usd {
            return false;
        }
        
        // Mock Portfolio check
        if proposal.amount_usd > (self.current_portfolio_value * 0.1) {
            return false;
        }

        true
    }

    fn check_solvent(&self, proposal: &TradeProposal) -> bool {
        // Mock simulation
        proposal.amount_in > 0.0
    }
}
