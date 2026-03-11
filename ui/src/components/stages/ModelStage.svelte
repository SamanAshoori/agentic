<script>
  import MetricCard from "../shared/MetricCard.svelte";
  import ConfirmButton from "../shared/ConfirmButton.svelte";
  import SectionCard from "../shared/SectionCard.svelte";
  import { confirmStage } from "../../api/client.js";

  let { data = null, status, onconfirmed, onrun, canRun = false } = $props();

  let modelType = $state("RandomForest");

  // RF params
  let nEstimators = $state(100);
  let maxDepth = $state(10);
  let classWeight = $state("balanced");
  let testSplit = $state(0.2);

  // LR params
  let maxIter = $state(1000);

  function handleTrain() {
    if (modelType === "LogisticRegression") {
      onrun({ model_type: "LogisticRegression", hyperparameters: { test_split: testSplit, max_iter: maxIter } });
    } else {
      onrun({ model_type: "RandomForest", hyperparameters: { n_estimators: nEstimators, max_depth: maxDepth, class_weight: classWeight, test_split: testSplit } });
    }
  }

  async function confirm() {
    await confirmStage("model", {
      model_type: modelType,
      hyperparameters: modelType === "LogisticRegression"
        ? { test_split: testSplit, max_iter: maxIter }
        : { n_estimators: nEstimators, max_depth: maxDepth, class_weight: classWeight },
    });
    onconfirmed();
  }

  let metrics = $derived(data?.model_metrics);
  let isLR = $derived(metrics?.model_type === "LogisticRegression" || (data && modelType === "LogisticRegression"));

  function pFmt(v) {
    if (v === null || v === undefined) return "—";
    if (v < 0.001) return "< .001";
    return v.toFixed(4);
  }
  function fFmt(v, d = 4) {
    if (v === null || v === undefined) return "—";
    return Number(v).toFixed(d);
  }
  function sigClass(v) {
    if (v === null || v === undefined) return "";
    if (v < 0.05) return "text-green-600 font-semibold";
    return "text-red-400";
  }
</script>

<div class="space-y-6">
  <!-- Configuration -->
  <SectionCard title="Model Configuration">
    {#snippet children()}
      <!-- Model type selector -->
      <div class="mb-4">
        <span class="block text-sm text-gray-500 mb-2">Model Type</span>
        <div class="flex gap-2">
          <button
            onclick={() => modelType = "RandomForest"}
            disabled={status === "confirmed"}
            class="px-4 py-2 rounded-lg text-sm font-medium transition-all border
                   {modelType === 'RandomForest'
                     ? 'bg-[var(--navy)] text-white border-[var(--navy)]'
                     : 'bg-white text-gray-600 border-gray-200 hover:border-gray-400'}"
          >Random Forest</button>
          <button
            onclick={() => modelType = "LogisticRegression"}
            disabled={status === "confirmed"}
            class="px-4 py-2 rounded-lg text-sm font-medium transition-all border
                   {modelType === 'LogisticRegression'
                     ? 'bg-[var(--navy)] text-white border-[var(--navy)]'
                     : 'bg-white text-gray-600 border-gray-200 hover:border-gray-400'}"
          >Logistic Regression</button>
        </div>
      </div>

      {#if modelType === "RandomForest"}
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <span class="block text-sm text-gray-500 mb-1">n_estimators</span>
            <input type="number" bind:value={nEstimators} disabled={status === "confirmed"}
              class="w-full px-3 py-2 rounded-lg border border-gray-200 text-sm focus:border-[var(--navy)] focus:outline-none" />
          </div>
          <div>
            <span class="block text-sm text-gray-500 mb-1">max_depth</span>
            <input type="number" bind:value={maxDepth} disabled={status === "confirmed"}
              class="w-full px-3 py-2 rounded-lg border border-gray-200 text-sm focus:border-[var(--navy)] focus:outline-none" />
          </div>
          <div>
            <span class="block text-sm text-gray-500 mb-1">test_split</span>
            <input type="number" bind:value={testSplit} step="0.05" min="0.1" max="0.5" disabled={status === "confirmed"}
              class="w-full px-3 py-2 rounded-lg border border-gray-200 text-sm focus:border-[var(--navy)] focus:outline-none" />
          </div>
          <div>
            <span class="block text-sm text-gray-500 mb-1">class_weight</span>
            <select bind:value={classWeight} disabled={status === "confirmed"}
              class="w-full px-3 py-2 rounded-lg border border-gray-200 text-sm focus:border-[var(--navy)] focus:outline-none">
              <option value="balanced">balanced</option>
              <option value="balanced_subsample">balanced_subsample</option>
            </select>
          </div>
        </div>
      {:else}
        <div class="grid grid-cols-2 gap-4">
          <div>
            <span class="block text-sm text-gray-500 mb-1">max_iter</span>
            <input type="number" bind:value={maxIter} step="100" min="100" max="5000" disabled={status === "confirmed"}
              class="w-full px-3 py-2 rounded-lg border border-gray-200 text-sm focus:border-[var(--navy)] focus:outline-none" />
          </div>
          <div>
            <span class="block text-sm text-gray-500 mb-1">test_split</span>
            <input type="number" bind:value={testSplit} step="0.05" min="0.1" max="0.5" disabled={status === "confirmed"}
              class="w-full px-3 py-2 rounded-lg border border-gray-200 text-sm focus:border-[var(--navy)] focus:outline-none" />
          </div>
        </div>
        <p class="mt-2 text-xs text-gray-400">Fits unregularized logistic regression. Outputs SPSS-style coefficient table with Wald statistics, odds ratios, and model fit indices.</p>
      {/if}

      {#if canRun && status === "pending"}
        <div class="mt-4">
          <button onclick={handleTrain}
            class="px-6 py-2.5 bg-[var(--accent)] text-[var(--navy)] font-semibold rounded-lg hover:bg-[#6bb8d3] transition-colors shadow-md">
            Train Model
          </button>
        </div>
      {/if}
    {/snippet}
  </SectionCard>

  <!-- Results -->
  {#if metrics}
    <!-- Top metrics -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <MetricCard label="ROC-AUC" value={fFmt(metrics.roc_auc)} />
      <MetricCard label="Target Recall" value={(metrics.target_recall * 100).toFixed(1) + "%"} />
      <MetricCard label="Precision (1)" value={(metrics.classification_report?.["1"]?.precision * 100).toFixed(1) + "%"} />
      <MetricCard label="Train / Val" value={`${metrics.train_shape?.[0]?.toLocaleString()} / ${metrics.val_shape?.[0]?.toLocaleString()}`} />
    </div>

    {#if metrics.model_type === "LogisticRegression"}
      <!-- Model Fit Statistics -->
      {#if metrics.model_fit}
        <SectionCard title="Model Fit Statistics">
          {#snippet children()}
            <div class="overflow-x-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr class="border-b border-gray-200 text-gray-500 text-xs uppercase tracking-wide">
                    <th class="pb-2 pr-4 text-left font-medium">−2 Log Likelihood</th>
                    <th class="pb-2 pr-4 text-right font-medium">AIC</th>
                    <th class="pb-2 pr-4 text-right font-medium">BIC</th>
                    <th class="pb-2 pr-4 text-right font-medium">Cox &amp; Snell R²</th>
                    <th class="pb-2 text-right font-medium">Nagelkerke R²</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td class="py-2 pr-4 font-mono">{fFmt(metrics.model_fit.neg2_log_likelihood, 3)}</td>
                    <td class="py-2 pr-4 font-mono text-right">{fFmt(metrics.model_fit.aic, 3)}</td>
                    <td class="py-2 pr-4 font-mono text-right">{fFmt(metrics.model_fit.bic, 3)}</td>
                    <td class="py-2 pr-4 font-mono text-right">{fFmt(metrics.model_fit.cox_snell_r2)}</td>
                    <td class="py-2 font-mono text-right">{fFmt(metrics.model_fit.nagelkerke_r2)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          {/snippet}
        </SectionCard>
      {/if}

      <!-- Coefficient Table -->
      {#if metrics.coefficients?.length > 0}
        <SectionCard title="Variables in the Equation">
          {#snippet children()}
            <div class="overflow-x-auto">
              <table class="w-full text-xs">
                <thead>
                  <tr class="border-b border-gray-200 text-gray-500 uppercase tracking-wide">
                    <th class="pb-2 pr-3 text-left font-medium">Variable</th>
                    <th class="pb-2 px-2 text-right font-medium">B</th>
                    <th class="pb-2 px-2 text-right font-medium">S.E.</th>
                    <th class="pb-2 px-2 text-right font-medium">Wald</th>
                    <th class="pb-2 px-2 text-right font-medium">df</th>
                    <th class="pb-2 px-2 text-right font-medium">Sig.</th>
                    <th class="pb-2 px-2 text-right font-medium">Exp(B)</th>
                    <th class="pb-2 px-2 text-right font-medium">95% CI Lower</th>
                    <th class="pb-2 text-right font-medium">95% CI Upper</th>
                  </tr>
                </thead>
                <tbody>
                  {#each metrics.coefficients as coef}
                    <tr class="border-b border-gray-50 hover:bg-gray-50">
                      <td class="py-1.5 pr-3 font-medium {coef.variable === '(Intercept)' ? 'text-gray-400 italic' : 'text-gray-800'}">{coef.variable}</td>
                      <td class="py-1.5 px-2 text-right font-mono">{fFmt(coef.B)}</td>
                      <td class="py-1.5 px-2 text-right font-mono text-gray-500">{fFmt(coef.SE)}</td>
                      <td class="py-1.5 px-2 text-right font-mono">{fFmt(coef.wald)}</td>
                      <td class="py-1.5 px-2 text-right text-gray-500">1</td>
                      <td class="py-1.5 px-2 text-right font-mono {sigClass(coef.p_value)}">{pFmt(coef.p_value)}</td>
                      <td class="py-1.5 px-2 text-right font-mono font-semibold">{fFmt(coef.exp_B)}</td>
                      <td class="py-1.5 px-2 text-right font-mono text-gray-500">{fFmt(coef.ci_lower_95)}</td>
                      <td class="py-1.5 text-right font-mono text-gray-500">{fFmt(coef.ci_upper_95)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
              <p class="text-xs text-gray-400 mt-2">Exp(B) = odds ratio. 95% CI on odds ratio scale. * p &lt; .05 &nbsp; ** p &lt; .01 &nbsp; *** p &lt; .001</p>
            </div>
          {/snippet}
        </SectionCard>
      {/if}

      <!-- Hosmer-Lemeshow -->
      {#if metrics.hosmer_lemeshow}
        <SectionCard title="Hosmer and Lemeshow Test">
          {#snippet children()}
            <div class="flex gap-6 text-sm">
              <div>
                <span class="text-gray-400 text-xs uppercase tracking-wide">Chi-square</span>
                <div class="font-mono font-semibold">{fFmt(metrics.hosmer_lemeshow.chi2)}</div>
              </div>
              <div>
                <span class="text-gray-400 text-xs uppercase tracking-wide">df</span>
                <div class="font-mono font-semibold">{metrics.hosmer_lemeshow.df}</div>
              </div>
              <div>
                <span class="text-gray-400 text-xs uppercase tracking-wide">Sig.</span>
                <div class="font-mono font-semibold {metrics.hosmer_lemeshow.p_value > 0.05 ? 'text-green-600' : 'text-red-500'}">{pFmt(metrics.hosmer_lemeshow.p_value)}</div>
              </div>
              <div>
                <span class="text-gray-400 text-xs uppercase tracking-wide">Result</span>
                <div class="font-medium {metrics.hosmer_lemeshow.p_value > 0.05 ? 'text-green-600' : 'text-red-500'}">{metrics.hosmer_lemeshow.interpretation}</div>
              </div>
            </div>
            <p class="text-xs text-gray-400 mt-2">A non-significant result (p &gt; .05) indicates good model fit.</p>
          {/snippet}
        </SectionCard>
      {/if}

      <!-- SPSS-style classification table -->
      {#if metrics.classification_table}
        <SectionCard title="Classification Table">
          {#snippet children()}
            {@const ct = metrics.classification_table}
            <div class="overflow-x-auto">
              <table class="text-sm w-auto">
                <thead>
                  <tr class="border-b border-gray-200">
                    <th class="pb-2 pr-8 text-left text-gray-500 font-medium">Observed</th>
                    <th class="pb-2 px-4 text-right text-gray-500 font-medium" colspan="2">Predicted</th>
                    <th class="pb-2 pl-4 text-right text-gray-500 font-medium">% Correct</th>
                  </tr>
                  <tr class="border-b border-gray-100">
                    <th class="pb-1"></th>
                    <th class="pb-1 px-4 text-right text-xs text-gray-400 font-medium">0</th>
                    <th class="pb-1 px-4 text-right text-xs text-gray-400 font-medium">1</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  <tr class="border-b border-gray-100 hover:bg-gray-50">
                    <td class="py-2 pr-8 font-medium">0</td>
                    <td class="py-2 px-4 text-right font-mono text-green-700">{ct.tn.toLocaleString()}</td>
                    <td class="py-2 px-4 text-right font-mono text-red-500">{ct.fp.toLocaleString()}</td>
                    <td class="py-2 pl-4 text-right font-semibold">{ct.pct_correct_0.toFixed(1)}%</td>
                  </tr>
                  <tr class="border-b border-gray-100 hover:bg-gray-50">
                    <td class="py-2 pr-8 font-medium">1</td>
                    <td class="py-2 px-4 text-right font-mono text-red-500">{ct.fn.toLocaleString()}</td>
                    <td class="py-2 px-4 text-right font-mono text-green-700">{ct.tp.toLocaleString()}</td>
                    <td class="py-2 pl-4 text-right font-semibold">{ct.pct_correct_1.toFixed(1)}%</td>
                  </tr>
                  <tr class="bg-gray-50">
                    <td class="py-2 pr-8 font-semibold">Overall %</td>
                    <td colspan="2"></td>
                    <td class="py-2 pl-4 text-right font-bold text-[var(--navy)]">{ct.overall_pct_correct.toFixed(1)}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          {/snippet}
        </SectionCard>
      {/if}

    {:else}
      <!-- Random Forest output (existing) -->
      {#if metrics.confusion_matrix}
        <SectionCard title="Validation Confusion Matrix">
          {#snippet children()}
            {@const cm = metrics.confusion_matrix}
            <div class="inline-grid grid-cols-2 gap-1 text-center">
              <div class="bg-green-50 rounded-lg p-4 min-w-[120px]">
                <div class="text-lg font-bold text-green-700">{cm[0][0]?.toLocaleString()}</div>
                <div class="text-xs text-green-600">True Neg</div>
              </div>
              <div class="bg-red-50 rounded-lg p-4 min-w-[120px]">
                <div class="text-lg font-bold text-red-700">{cm[0][1]?.toLocaleString()}</div>
                <div class="text-xs text-red-600">False Pos</div>
              </div>
              <div class="bg-red-50 rounded-lg p-4 min-w-[120px]">
                <div class="text-lg font-bold text-red-700">{cm[1][0]?.toLocaleString()}</div>
                <div class="text-xs text-red-600">False Neg</div>
              </div>
              <div class="bg-green-50 rounded-lg p-4 min-w-[120px]">
                <div class="text-lg font-bold text-green-700">{cm[1][1]?.toLocaleString()}</div>
                <div class="text-xs text-green-600">True Pos</div>
              </div>
            </div>
          {/snippet}
        </SectionCard>
      {/if}

      {#if metrics.feature_importances}
        <SectionCard title="Feature Importances">
          {#snippet children()}
            <div class="space-y-2">
              {#each Object.entries(metrics.feature_importances) as [feat, imp]}
                <div class="flex items-center gap-3 text-sm">
                  <span class="w-28 text-right text-gray-600 truncate">{feat}</span>
                  <div class="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
                    <div class="h-full rounded-full bg-[var(--navy)]"
                         style="width: {imp * 100 / Object.values(metrics.feature_importances)[0] * 100}%"></div>
                  </div>
                  <span class="w-14 text-xs text-gray-400">{(imp * 100).toFixed(1)}%</span>
                </div>
              {/each}
            </div>
          {/snippet}
        </SectionCard>
      {/if}
    {/if}

    {#if status === "awaiting_review"}
      <div class="flex justify-end">
        <ConfirmButton onclick={confirm} />
      </div>
    {/if}
  {/if}
</div>
