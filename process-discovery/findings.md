# Findings

Here are the collected findings from our analysis of the event log and the subsequent process discovery.

## 1. Data Analysis & Event Log Insights

Before discovering any process, we first analyzed the event log itself. This revealed key characteristics of the data that guided our modeling approach.

**Log Size and Complexity:** The initial event log is massive, containing **319,233 cases (traces)**. This volume, combined with a high number of process variants (3,427 variants in a filtered log), indicates that a simple, one-shot process discovery would result in an unreadable "spaghetti model."

**Log Scope Limitation (Missing Payment):** A key finding is that the event log is incomplete. While it covers the process from requisition or purchase order creation through invoice receipt, the actual **financial settlement step (i.e., "Make Payment") is missing**. The process most often ends with "Clear Invoice" (9258 times), which is an internal accounting step, not the final payment to the vendor.

**Logging of External Events:** The activity **"Vendor creates invoice"** is an external business event, but it's captured in the log. This is probably because the internal system records the *registration* of that event (e.g., via an EDI document or a manual entry like SAP's MIRO transaction). This is expected behavior and represents the system's "awareness" of the external action.

## 2. Process Discovery

Our primary challenge was to reduce the complexity of the log into a simple, generalizable, and accurate process model. This required a multi-step process (see jupyter notebooks for details).

+ **Initial Model testing (DFG & Heuristic Miner):**
     * The first pass using a **Directly-Follows Graph (DFG)** directly on the unfiltered data resulted in a "spaghetti process".
    * A **Heuristic Miner** was also attempted but also showed too many infrequent pathways.
    * The default **Inductive Miner** algorithm was also not suitable for the same reason

    * The produced models had potentially **high fitness** but had **poor generalization and simplicity**. This is because the default settings try to account for every single pathway in data, which is undesirable with such a noisy log.

* **Strategy 1: Noise Threshold:**
    * Our first attempt at simplification was using the Inductive Miner with a `noise_threshold=0.4`.
    * While this filtered some noise, the resulting model was still complex, indicating that simple noise-based filtering was insufficient.

* **Strategy 2: Aggressive Filtering:**
    * To find the *main* process, we adopted a more direct filtering strategy.
    * **1. Filter by Start/End Activities:** We kept only cases that started and ended with the most common activities (those representing >= 5% of all cases). This filtered the log from 15,182 cases down to 13,098.
    * **2. Filter by Variant Coverage:** We then filtered this log to keep only the most frequent variants that covered **10% of the remaining cases**. This was a significant reduction, leaving only **3,715 cases**.
    * **Result:** This "happy path" log produced a very clean, simple BPMN model showing the core process.While this model is simple, much of the information is lost and around 9,000 cases were filtered out.


 **Isolating & Refining:** 
 \
 We then adopted a "divide and conquer" approach:
* **SRM Subprocess:** We filtered the log to *only* contain events with "SRM" in their name. This isolated log was then mined to create a clean model of just the SRM subprocess. We semantically excluded rare deviations like `SRM: Incomplete` and `SRM: Held` (5 cases each) as noise.
* **Main Process:** We returned to a more balanced filter of the main log (e.g., 1% variant coverage, leaving 7,199 cases). From this mined model, we applied domain knowledge of the P2P process and **removed `Change Price` and `Change Quantity`**. We reasoned these are not core P2P flow activities but rather exceptions that should be handled in the requisition phase. Their low frequency (172 and 131 cases, respectively, out of 7199) supported this decision.

4.  **Final Combined Model:** Finally, we **manually combined** the refined main process model and the isolated SRM subprocess model into a single, BPMN diagram. This final model was built trying to balance the data log and semantic process logic.



## 3. Key Process Insights & Deviations

This structured discovery approach, combined with analysis of the unfiltered log, revealed some insights about how the business *actually* operates.

### Insight 1: Widespread "Maverick Buying"
In a standard Purchase-to-Pay (P2P) process, every purchase should begin with a Purchase Requisition (PR).

* **Expectation:** The process should start with `Create Purchase Requisition Item`.
* **Reality:** `Create Purchase Order Item` is the starting activity in **13,035 cases**, while `Create Purchase Requisition Item` is the start in only **1,219 cases**
* **Conclusion:** This strongly suggests "maverick buying," where employees are creating purchase orders directly without a formal, approved requisition. This indicates a deviation from the expected process in Purchase-to-Pay.

### Insight 2: The Isolated "SRM" Subprocess
Our analysis of start activities also revealed that **564 cases begin with `SRM: Created`**.

* **Finding:** Further investigation showed that SRM (Supplier Relationship Management) activities form a self-contained process.

* **Deviations Ignored:** Rare deviations like `SRM: Incomplete` and `SRM: Held` were noted but occurred in only 5 cases and were considered noise for the main subprocess model.

* **Integration Gaps:** We also found that cases involving `SRM: Transfer Failed (E.Sys.)` (45 occurrences) just end abruptly. This suggests a potential **data handover or system integration failure** between the SRM system and the main P2P execution system.

### Insight 3: Iterative Deliveries vs. Final Invoices


* **Observation:** The log shows frequent self-loops on activities related to delivery, but not on invoicing.
* **Data:**
    * `Record Service Entry Sheet` loops on itself **139,107 times**.
    * `Record Goods Receipt` loops on itself **40,624 times**.
    * `Record Invoice Receipt` loops on itself only **3,579 times** with an absolute frequency of **18937**.

* **Explanation (Quoted from Case Study):**
    > For a single purchase order item, there can be many goods
receipt documents and corresponding invoices, which are subsequently paid. Consider for example that
an IT department of a university purchases ten laptops for ten different chairs; the vendor has five
laptops on stock and delivers them right away, but has to reorder the other five which are shipped in
another delivery. Hence, there is one purchase order item, two goods receipts, and (possibly) ten
invoices (one per chair that has to pay their own laptop).

* **Conclusion:** This data reflects the operational reality of partial or staggered deliveries.
    1. **Service Procurement (SES):** This is highly iterative, with services frequently delivered and confirmed in multiple phases.
    2. **Material Procurement (GR):** This is also iterative, but less so than services. It reflects partial deliveries of physical goods.
    3.  **Invoicing (IR):** This step is more "final." Invoices are generally recorded once per delivery or phase, not in rapid, repetitive succession.