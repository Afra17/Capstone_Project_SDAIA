# فَــزْعَــة (Fazaa)  
### Automated RFP–Proposal Comparison System


**Team Members:**  
- Afra Aloufi  
- Amnah Albrahim  
- Luluh Almogbil  

---

## Project Overview

**فَــزْعَــة** is an AI-powered assistant designed to simplify how government entities evaluate vendor proposals.

RFP reviews are usually manual, slow, and inconsistent. Our system helps teams extract requirements, compare submissions, and score vendors automatically — enabling faster and more objective decisions.

---

## Problem Statement

Evaluating multiple long proposals against complex RFPs leads to:

- Long review times  
- Human errors  
- Inconsistent scoring  
- Compliance risks  
- Delayed decisions  

---

## Proposed Solution

فَــزْعَــة uses a multi-agent AI system to:

- Extract RFP requirements  
- Understand vendor responses  
- Compare both sides  
- Score vendors  
- Produce a final ranking  

---

##  Phase 1 — Data Collection

To build and test the first version of the system:

- One real RFP was obtained from confidential resources.  
- Three vendor proposals were generated using an LLM by **Luluh Almogbil**.  
- This proposal generation happened **in parallel** with Afra Aloufi working on Agents 1 & 2 and Amnah Ibrahim working on Agent 3.

---

##  High-Level Flow

Upload RFP → Upload Proposals → Extract → Compare → Score → Rank.

---

##  Multi-Agent Setup

| Agent | Role | Responsible |
|------|------|-----------|
| Agent 1 | Extracts main RFP goals & requirements | Afra Aloufi |
| Agent 2 | Adds context and compliance details | Afra Aloufi |
| Agent 3 | Extracts and structures proposal responses | Amnah Albrahim |
| Agent 4 *(Current)* | Core evaluation engine (technical & financial sub-agents) | Luluh Almogbil |
| Agent 5 *(Planned)* | Final ranking & reasoning | To be determined |

---


---
