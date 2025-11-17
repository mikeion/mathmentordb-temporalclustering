# Power User Analysis - MathConverse Dataset

## Summary Statistics

**Dataset**: 184,539 conversations (after filtering to ≥3 messages)

### Power Users Defined as ≥10 Conversations

| Role | Total Individuals | Power Users | % Power Users | Conversations Covered |
|------|-------------------|-------------|---------------|----------------------|
| **Tutors** | 22,554 | 2,308 | 10.3% | 275,851 (149.5%*) |
| **Students** | 36,682 | 3,728 | 10.6% | 111,743 (60.6%) |

*Note: >100% because some conversations involve multiple power user tutors

### Top 10 Most Active Tutors

1. **261933205387477002** (Nicolas Miller): 12,332 conversations, 56,504 messages
2. 183668144404037632: 9,216 conversations
3. 220517274845577216: 7,173 conversations
4. 485903210272391178: 6,665 conversations
5. 456226577798135808: 5,695 conversations
6. 182179228840755200: 4,084 conversations
7. 132304137583984641: 4,013 conversations
8. 349354952961032192: 4,011 conversations
9. 186109587366215691: 3,701 conversations
10. 187850624421986304: 3,043 conversations

### Top 10 Most Active Students

1. **456226577798135808**: 6,851 conversations (also appears as tutor!)
2. 187850624421986304: 734 conversations
3. 819818586965016596: 623 conversations
4. 313298495417221121: 522 conversations
5. 692635975678951464: 459 conversations
6. 849961886874730516: 346 conversations
7. 349354952961032192: 325 conversations
8. 485903210272391178: 320 conversations
9. 261933205387477002: 316 conversations
10. 437300702960812033: 311 conversations

## Deep Dive: Top Power User (Nicolas Miller)

- **Author ID**: 261933205387477002
- **Total conversations**: 12,332
- **Total messages**: 56,504 (54,996 as tutor, 1,508 as student)
- **Active period**: Jan 15, 2022 - Feb 4, 2023 (384 days)
- **Messages per day**: 147.1
- **Conversations per day**: 32.1

### Sample Messages

```
"That's one definition of the difference quotient"
"This is the one for h going to zero. The other is x going to 3."
"Why are you using h at all?"
"Plug in the definition of f in this expression and you'll get your answer"
"$$[(x+5)/(x+1) - 2] / (x-3)$$"
"Did you simplify?"
"You should get a cancellation"
"a/b/c is the same as a/(bc)"
"Do you see the cancellation"
"I would put extra parentheses to be careful"
```

### Assessment: Bot or Human?

**Likely human, but extremely active**:
- 147 messages/day = ~18 msgs/hour over 8-hour workday
- Messages are pedagogically sound, contextual
- Shows typical tutoring patterns (questions, hints, explanations)
- But volume is extraordinarily high

**Red flags for automation**:
- Sustained high volume (>30 conversations/day for a full year)
- No obvious breaks or days off
- Could be a dedicated platform moderator or "super tutor"

**Recommendation**: 
- Include in power user analysis BUT
- Consider sensitivity analysis excluding top 1% of users
- Check if excluding changes temporal patterns significantly

## Research Implications

### 1. Hierarchical Modeling Opportunity

**6,036 power users** (2,308 tutors + 3,728 students) represent only **10.5% of individuals** but participate in **>60% of conversations**.

This creates perfect conditions for selective hierarchical modeling:

```python
# Estimate individual effects ONLY for power users
theta_conv = (
    mu_cluster[cluster_id] +
    sigma_tutor * z_tutor[tutor_id]  # if power user
    + sigma_student * z_student[student_id]  # if power user
)
```

Benefits:
- Manageable computational cost (6k vs 60k parameters)
- Reliable estimates (≥10 observations per individual)
- Covers majority of data

### 2. Power Law Distribution

Both tutors and students follow power-law distributions:

**Tutors**:
- Median: 1 conversation
- 75th percentile: 3 conversations
- 90th percentile: 10 conversations
- 99th percentile: 195 conversations
- Max: 12,332 conversations

**Students**:
- Median: 2 conversations
- 75th percentile: 4 conversations
- 90th percentile: 10 conversations
- 99th percentile: 58 conversations
- Max: 7,364 conversations

This suggests:
- Most users are "one-time" or "casual" users
- Small fraction drives platform activity
- Classic "1% rule" of online participation

### 3. Cross-Role Participation

**Key finding**: Some users appear as BOTH tutors and students!

Examples:
- 456226577798135808: Top tutor (#5 with 5,695 convs) AND top student (#1 with 6,851 convs)
- 261933205387477002 (Nicolas Miller): Top tutor (#1) but also 316 student conversations
- 349354952961032192: Active as both tutor and student

This enables studying:
- How tutoring experience affects learning behavior
- Do good tutors ask different kinds of questions as students?
- Temporal patterns when same person switches roles

### 4. Temporal Consistency Research Questions

**RQ1**: Do power users have more consistent temporal patterns across conversations?
- **H1**: High-volume tutors develop stable pacing "signatures"
- **H2**: Power users show less variance in BC, CD than casual users

**RQ2**: Do power users adapt their pacing to students, or impose their style?
- **H1**: Novice tutors (few conversations) → high individual effect (impose style)
- **H2**: Expert tutors (many conversations) → low individual effect (adapt to students)

**RQ3**: Does power user temporal style predict student outcomes?
- **H1**: Students paired with "consistent pace" tutors show better completion rates
- **H2**: Matching student/tutor pacing styles predicts engagement

## Next Steps

1. **Validate top users**: Manually inspect top 10 tutors/students for bot-like behavior
2. **Sensitivity analysis**: Rerun clustering with/without top 1% users
3. **Power user profiling**: Extract temporal signatures for frequent users
4. **Cross-role analysis**: Study users who appear as both tutor and student
5. **Hierarchical modeling**: Implement selective individual effects for power users only

## Data Quality Considerations

**Potential issues**:
- Top user could be automated or semi-automated
- Platform structure might encourage power user behavior
- Need to distinguish: dedicated humans vs bots vs multiple people sharing account

**Validation needed**:
- Check message timing patterns (do they work 24/7 or have breaks?)
- Examine response latencies (instant = bot-like)
- Compare message content diversity (copy-paste vs original)

