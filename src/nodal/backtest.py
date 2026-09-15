"""Cost-aware, walk-forward backtests for the inefficiency scans.

Every strategy is evaluated the same way:

    * walk-forward: parameters fit on window t, traded on window t+1, rolled
    * costs: per-MWh transaction cost + ERCOT uplift estimate for virtuals;
      spread trades pay both legs
    * capacity honesty: nodal virtual markets are thin; sizing is capped and
      results reported per MW, not scaled to fantasy notional
    * reporting: net P&L series, Sharpe (hourly -> annualized correctly),
      max drawdown, hit rate, and the share of P&L from the top 1% of hours
      (tail dependence is the real risk story in power)

Planned API:
    virtual_backtest(dart_signal, dart_realized, cost) -> BacktestResult
    spread_backtest(zscores, spreads, entry, exit, cost) -> BacktestResult
"""
