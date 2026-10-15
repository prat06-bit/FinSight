# FinSight Knowledge Base

FinSight analyzes yearly company financial data from CSV files. It focuses on profitability, liquidity, leverage, cash generation, and growth.

Net margin compares net income with revenue. A falling net margin can show pricing pressure, rising costs, higher interest expense, or weaker operating discipline.

Current ratio measures liquidity by comparing current assets with current liabilities. Values above 1.0 suggest the company has enough short-term assets to cover short-term obligations.

Debt to equity compares total debt with shareholder equity. A rising value means leverage is increasing and the business may become more sensitive to interest costs.

Free cash flow is operating cash flow minus capital expenditures. Positive free cash flow means the company generates cash after reinvesting in the business.

Interest coverage compares operating income with interest expense. Higher coverage means the company has more operating profit available to service debt.

The ML risk score in this project is a transparent baseline model. It is not trained on external market data yet; it uses financial thresholds and recent trends.

The MCP layer exposes FinSight actions as machine-readable tools so another agent or client can request reports, risk scoring, forecasts, and local retrieval answers.
