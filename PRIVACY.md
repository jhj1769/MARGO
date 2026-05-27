# MARGO Research Privacy Policy

**Last updated**: 2026-05-25
**Project**: MARGO — A Multi-Agent Framework for Stakeholder-Aware Recommendation Governance
**Affiliation**: POSTECH AIM Lab, Pohang University of Science and Technology

---

This is an academic research project investigating LLM-based multi-agent
recommendation governance for the fashion domain. This Privacy Policy
describes how the MARGO research codebase handles data accessed from
third-party APIs (Pinterest, Google Trends, YouTube, GDELT) used as
input signals to our trend analysis pipeline.

## Data We Access

We access only **public aggregate data** from third-party platforms for
the purpose of building trend signal snapshots:

- **Pinterest Trends API**: aggregate keyword-level trend signals
  (weekly time series, 0–100 scale) via the `/v5/trends/keywords/{region}/top/{trend_type}` endpoint
- **Google Trends**: aggregate keyword search interest
- **YouTube Data API**: aggregate keyword mention counts in public video metadata
- **GDELT 2.0 GKG**: public fashion-trade-press article metadata

## Data We DO NOT Collect

- We do **NOT** collect any personally identifying information (PII)
- We do **NOT** collect individual user behavior, posts, pins, or boards
- We do **NOT** track individual users
- We do **NOT** use cookies, analytics, or third-party tracking
- We do **NOT** collect images, videos, or any user-generated content

## How We Use Data

- Aggregate trend keyword data is used **exclusively** as input signal
  to research models within our academic framework (MARGO)
- Data is stored locally in our research environment for reproducibility
  of published academic results
- Data is **NOT** redistributed to third parties
- Data is **NOT** used for advertising, commercial purposes, or any
  monetized service
- Data is **NOT** used to train commercial machine learning or AI models
  beyond the scope of this academic research project

## Data Retention

Trend snapshots are retained locally only for the duration necessary to
reproduce published academic results. We do not maintain a long-term
public database of accessed data.

## Compliance

This research project complies with:
- Pinterest Developer Terms and Data API Terms
- Google API Services User Data Policy
- YouTube API Services Terms of Service
- GDELT 2.0 Project Terms of Use
- POSTECH research ethics guidelines

## Contact

For questions regarding this Privacy Policy or the research project:

**Email**: hjjung@analyticsim.org
**Affiliation**: POSTECH AIM Lab
**Repository**: https://github.com/jhj1769/MARGO

---

*This Privacy Policy applies solely to the academic MARGO research
codebase. It does not cover any commercial product, service, or
deployment.*
