"""Locational spread pairs: stationary node-minus-node spreads.

Two nodes with similar constraint exposure should have a spread that mean-
reverts; a spread that is cointegrated with a short half-life is a candidate
inefficiency. Pipeline:

    1. candidates   restrict to pairs close in factor-loading space (avoids
                    testing ~600k arbitrary pairs; also a physical prior)
    2. test         Engle-Granger cointegration on the in-sample window
    3. dynamics     fit OU process to the spread -> half-life, equilibrium, sigma
    4. signal       z-score of the spread vs its rolling equilibrium
    5. validate     out-of-sample: does reversion persist? survives BH across
                    all candidates tested?

Planned API:
    candidate_pairs(loadings, max_distance) -> list[(a, b)]
    test_pair(spread) -> dict(coint_p, half_life, sigma)
    scan(basis, candidates) -> DataFrame with BH-corrected discoveries
"""
