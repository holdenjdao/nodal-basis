"""Node profiles: one row per settlement point summarizing its personality.

A node is an electrical location. Its price is hub energy plus the shadow
prices of whichever transmission constraints it is exposed to (weighted by
its shift factors) plus losses. Everything we measure about a node is
therefore a statement about its exposure to constraints:

    level        mean basis (structurally cheap / rich vs hub)
    dispersion   basis std, mean |basis|, skew, tail frequency (spike share)
    shape        hour-of-day basis profile (solar nodes sag midday, etc.)
    exposure     factor loadings (which constraint axes move it)
    residual     variance unexplained by the factor model (idiosyncratic risk)
    dart         DA-RT premium stats from `dart.py`

Planned API:
    profile_table(basis, factor_model, dart_table) -> DataFrame indexed by location
    similar_nodes(profiles, location, k) -> nearest neighbours in exposure space
"""
