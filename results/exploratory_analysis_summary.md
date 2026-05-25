# Exploratory Analysis Summary

This note closes the exploratory analysis task of the Barents Watch lice case with concrete answers from the prepared master table.

Method used:

- Historical scope: full 2012-2026 exploratory coverage from the prepared master table built on top of the raw CSV files.
- The currently available 2026 data is partial through 2026-05-18, so descriptive summaries include the latest 2026 weeks where available but late-year seasonality is still driven mostly by prior years.
- Active site-weeks: `likelynofish == False`.
- Active counted site-weeks: `likelynofish == False` and `havecountedlice == True`.
- Breach calculations: `breach_this_week` from the processed master table.
- Production-area names were canonicalized by `productionareaid` because raw labels contain spelling variants such as `Ryfylke` and `Ryfylket`.

## 1. Which production areas have the most treatments relative to the number of active sites?

Measured as treatment events per 100 active site-weeks.

| Rank | Production area | Treatment events per 100 active site-weeks | Treated site-week rate |
| --- | --- | ---: | ---: |
| 1 | Svenskegrensen til Jæren | 23.74 | 16.9% |
| 2 | Stadt til Hustadvika | 21.43 | 15.6% |
| 3 | Karmøy til Sotra | 19.63 | 15.5% |
| 4 | Nord-Trøndelag med Bindal | 18.99 | 16.7% |
| 5 | Nordmøre og Sør-Trøndelag | 18.95 | 15.0% |
| 6 | Nordhordland til Stadt | 17.68 | 14.7% |
| 7 | Ryfylket | 16.57 | 12.6% |

Lowest treatment intensity:

- Øst-Finnmark: 2.42 treatment events per 100 active site-weeks.
- Vest-Finnmark: 5.64.
- Kvaløya til Loppa: 6.05.

Interpretation:

- Treatment intensity is clearly highest in the south and west.
- The western hotspot corridor from Karmøy north through Hustadvika combines both high treatment volume and high breach pressure, which suggests treatments are responding to genuinely difficult lice conditions rather than replacing them.

## 2. Which regions or production areas breach lice limits most consistently?

Measured as breach rate on active counted site-weeks, plus the share of weeks in which the production area had at least one breached site.

| Rank | Production area | Breach rate | Share of weeks with any breach |
| --- | --- | ---: | ---: |
| 1 | Stadt til Hustadvika | 6.06% | 66.8% |
| 2 | Karmøy til Sotra | 5.72% | 88.8% |
| 3 | Nordhordland til Stadt | 5.23% | 83.3% |
| 4 | Nordmøre og Sør-Trøndelag | 4.83% | 86.3% |
| 5 | Nord-Trøndelag med Bindal | 4.53% | 50.3% |
| 6 | Helgeland til Bodø | 4.44% | 60.7% |

Lowest breach pressure:

- Øst-Finnmark: 0.44% breach rate.
- Svenskegrensen til Jæren: 1.22%.
- Kvaløya til Loppa: 1.53%.

Regional county hotspots among counties with at least 5,000 active counted site-weeks:

- Nord-Trøndelag: 9.15% breach rate.
- Sogn og Fjordane: 8.02%.
- Hordaland: 7.44%.

Interpretation:

- The strongest and most persistent breach corridor runs through western and mid-Norway.
- Karmøy til Sotra and Nordmøre og Sør-Trøndelag stand out not only for high breach rates, but also because they have some breach activity in nearly every year-round cycle.

## 3. At which temperatures does lice pressure appear most prevalent in each production area?

Temperatures were summarized in 2 C bands using active counted site-weeks with at least 200 observations per production area-band. The table below reports the band with the highest mean female-adult-to-limit ratio, with the highest-breach band noted where it differs. Including the available 2026 weeks does not materially change the broad temperature story.

| Production area | Peak pressure band | Peak breach band | Support site-weeks |
| --- | --- | --- | ---: |
| Svenskegrensen til Jæren | 16-18 C | 16-18 C | 572 |
| Ryfylket | 18-22 C | 18-22 C | 226 |
| Karmøy til Sotra | 18-22 C | 18-22 C | 678 |
| Nordhordland til Stadt | 10-12 C | 10-12 C | 8,287 |
| Stadt til Hustadvika | 14-16 C | 14-16 C | 3,006 |
| Nordmøre og Sør-Trøndelag | 16-18 C | 16-18 C | 469 |
| Nord-Trøndelag med Bindal | 14-16 C | 12-14 C | 1,482 |
| Helgeland til Bodø | 10-12 C | 10-12 C | 6,191 |
| Vestfjorden og Vesterålen | 16-18 C | 16-18 C | 304 |
| Andøya til Senja | 14-16 C | 14-16 C | 542 |
| Kvaløya til Loppa | 12-14 C | 12-14 C | 672 |
| Vest-Finnmark | 12-14 C | 12-14 C | 704 |
| Øst-Finnmark | 6-8 C | 6-8 C | 380 |

Interpretation:

- Lice pressure generally peaks in warm late-summer or early-autumn water, but the exact band shifts northward.
- Southern and western areas peak around 14-18 C, with the warmest southern areas still elevated above 18 C.
- Northern areas peak earlier on the temperature scale, mostly around 10-14 C, and Øst-Finnmark peaks lower at 6-8 C because its seasonal temperature range is colder overall.

## 4. Are there seasonal, geographical, or treatment-related patterns worth highlighting?

Seasonality:

- National breach rates bottom out in March to June at roughly 2.5% to 3.1%.
- Breach pressure rises sharply from July and peaks in September at 7.72%, then stays elevated in October at 7.00%.
- Treatment activity follows the same pattern: 20.1% of active counted site-weeks have any treatment in August and 20.8% in September.
- Because 2026 currently only covers the first part of the year, the month-by-month pattern should be read as full-history seasonality plus the latest early-2026 observations, not as a full new annual cycle.

Hotspot seasonality by production area:

- Nordmøre og Sør-Trøndelag peaks in September at 9.76% breach rate.
- Nord-Trøndelag med Bindal peaks in September at 9.43%.
- Helgeland til Bodø peaks in September at 10.93%.
- Vestfjorden og Vesterålen peaks in September at 10.05%.
- Stadt til Hustadvika is high across August to October and also shows a winter spike in January.
- Karmøy til Sotra is unusual: January is its single highest breach month at 8.15%, with another strong period in July to August.

Geography:

- The two middle latitude quartiles have the highest breach rates: 5.76% and 4.35%.
- The far north has materially lower treatment intensity and lower breach rates, which is consistent with colder water and lower observed lice pressure.

Treatment-related patterns:

- Same-week treatment is more common during breach weeks than non-breach weeks: 20.8% versus 12.4%.
- The week after a breach, treatment probability jumps to 38.2% versus 11.6% after non-breach weeks.
- Sites that breach have also seen heavier recent intervention: average treatments in the prior 4 weeks are 0.707 for breached site-weeks versus 0.501 for non-breached site-weeks.

Interpretation:

- Treatments appear reactive rather than fully preventive. They increase around breaches, but the highest-pressure areas still accumulate repeated breaches despite heavier intervention.

## 5. What other interesting biological, operational, or regional insights can be found?

- Breaches are concentrated. 1,067 sites have at least one historical breach week, and the top 10% of those sites account for 31.7% of all breach weeks.
- Several repeat-breach sites come from the same western and mid-Norway hotspot areas, including Sauaneset I, Urda, Stokkvika, Kråkåsen, and Oslandsurda. Sauaneset I has to 282 historical breach weeks and Sandnesbukta has 128.
- The treatment map and breach map are aligned geographically. High-treatment areas are not the same as low-risk areas; they are often the same difficult operating regions.
- Medicinal treatments are the largest broad treatment category in the treatment table at 27,709 aggregated events, followed by mechanical treatments at 19,523, while cleanerfish use is also common at 17,066. That suggests operators are already using multiple intervention modes in the hardest areas.

## 6. What is the correlation between adult female lice and mobile lice in each production area?

Pearson correlation on active counted site-weeks with non-null adult female and mobile lice counts.

| Production area | Correlation |
| --- | ---: |
| Vest-Finnmark | 0.689 |
| Nord-Trøndelag med Bindal | 0.678 |
| Svenskegrensen til Jæren | 0.654 |
| Kvaløya til Loppa | 0.623 |
| Vestfjorden og Vesterålen | 0.614 |
| Andøya til Senja | 0.610 |
| Nordmøre og Sør-Trøndelag | 0.590 |
| Karmøy til Sotra | 0.563 |
| Helgeland til Bodø | 0.558 |
| Stadt til Hustadvika | 0.551 |
| Ryfylket | 0.510 |
| Nordhordland til Stadt | 0.462 |
| Øst-Finnmark | 0.387 |

Interpretation:

- The correlation is positive everywhere, but its strength varies materially by area.
- The relationship is strongest in the north, where adult and mobile lice appear to move together more tightly.
- Western areas such as Nordhordland til Stadt show weaker correlation, which can indicate more complex treatment timing, cohort structure, or local dynamics between lice stages.

## Presentation-ready bottom line

- The main biological and operational hotspot runs from Karmøy north through Hustadvika and into Trøndelag.
- National pressure is strongly seasonal, peaking in late summer and early autumn.
- A relatively small group of repeat-breach sites accounts for a disproportionate share of total breach weeks.