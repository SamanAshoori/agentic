<script>
  import PipelineStepper from "./components/PipelineStepper.svelte";
  import ETLStage from "./components/stages/ETLStage.svelte";
  import StatsStage from "./components/stages/StatsStage.svelte";
  import ModelStage from "./components/stages/ModelStage.svelte";
  import EvaluateStage from "./components/stages/EvaluateStage.svelte";
  import StatusBadge from "./components/shared/StatusBadge.svelte";
  import {
    getStatus,
    uploadDataset,
    runStats,
    runModel,
    runLogistic,
    runEvaluate,
    resetSession,
    downloadReport,
  } from "./api/client.js";

  const STAGES = ["etl", "stats", "model", "evaluate"];
  const stageTitles = {
    etl: "Column Review & Cleaning",
    stats: "Feature Selection",
    model: "Model Training",
    evaluate: "Evaluation",
  };

  let stageStatus = $state({ etl: "pending", stats: "pending", model: "pending", evaluate: "pending" });
  let confirmedOutputs = $state({});
  let activeStage = $state("etl");
  let stageData = $state({});
  let datasetUploaded = $state(false);
  let datasetInfo = $state(null);
  let error = $state(null);
  let running = $state(false);

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    error = null;
    try {
      datasetInfo = await uploadDataset(file);
      datasetUploaded = true;
      stageStatus = { etl: "pending", stats: "pending", model: "pending", evaluate: "pending" };
      stageData = {};
      confirmedOutputs = {};
    } catch (e) {
      error = e.message;
    }
  }

  async function handleRunStats() {
    error = null;
    running = true;
    try {
      const result = await runStats();
      stageData.stats = result;
      stageStatus.stats = "awaiting_review";
    } catch (e) {
      error = e.message;
    }
    running = false;
  }

  async function handleRunModel({ model_type, hyperparameters }) {
    error = null;
    running = true;
    try {
      const result = model_type === "LogisticRegression"
        ? await runLogistic({ model_type, hyperparameters })
        : await runModel({ model_type: "RandomForest", hyperparameters });
      stageData.model = result;
      stageStatus.model = "awaiting_review";
    } catch (e) {
      error = e.message;
    }
    running = false;
  }

  async function handleRunEvaluate() {
    error = null;
    running = true;
    try {
      const result = await runEvaluate();
      stageData.evaluate = result;
      stageStatus.evaluate = "awaiting_review";
    } catch (e) {
      error = e.message;
    }
    running = false;
  }

  function handleConfirmed() {
    getStatus().then((s) => {
      stageStatus = s.stage_status;
      confirmedOutputs = s.confirmed_outputs;
      const idx = STAGES.indexOf(activeStage);
      if (idx < STAGES.length - 1) {
        activeStage = STAGES[idx + 1];
      }
    });
  }

  async function handleNewRun() {
    await resetSession();
    datasetUploaded = false;
    datasetInfo = null;
    stageData = {};
    stageStatus = { etl: "pending", stats: "pending", model: "pending", evaluate: "pending" };
    confirmedOutputs = {};
    activeStage = "etl";
    error = null;
    running = false;
  }

  // Can run stats/evaluate (no payload needed). Model is triggered from its component.
  let canRunActive = $derived(() => {
    if (activeStage === "etl") return false; // manual stage
    if (activeStage === "model") return false; // triggered from component with hyperparams
    if (running) return false;
    const s = stageStatus[activeStage];
    if (s === "running" || s === "awaiting_review" || s === "confirmed" || s === "complete") return false;
    const idx = STAGES.indexOf(activeStage);
    const prev = STAGES[idx - 1];
    return stageStatus[prev] === "confirmed" || stageStatus[prev] === "complete";
  });

  function handleRunActive() {
    if (activeStage === "stats") handleRunStats();
    else if (activeStage === "evaluate") handleRunEvaluate();
  }
</script>

<div class="min-h-screen flex flex-col">
  <header class="bg-[var(--navy)] text-white px-6 py-4 flex items-center justify-between shadow-lg">
    <div>
      <h1 class="text-xl font-bold tracking-tight">ML Pipeline</h1>
      <p class="text-sm text-[var(--ice-blue)] opacity-80">Dataset-agnostic classification</p>
    </div>
    <div class="flex items-center gap-3">
      {#if datasetUploaded}
        <span class="text-sm text-[var(--ice-blue)]">{datasetInfo?.filename} &mdash; {datasetInfo?.rows?.toLocaleString()} rows</span>
      {/if}
      <button
        onclick={handleNewRun}
        class="px-4 py-1.5 text-sm rounded-lg border border-white/30 hover:bg-white/10 transition-colors"
      >
        New Run
      </button>
      <button
        onclick={downloadReport}
        class="px-4 py-1.5 text-sm rounded-lg border border-white/30 hover:bg-white/10 transition-colors"
      >
        Download Report
      </button>
    </div>
  </header>

  <div class="flex flex-1 overflow-hidden">
    <aside class="w-72 bg-[#f0f2f8] p-4 overflow-y-auto border-r border-gray-200 shrink-0">
      <PipelineStepper
        stages={STAGES}
        {stageStatus}
        {activeStage}
        onselect={(s) => activeStage = s}
      />
    </aside>

    <main class="flex-1 overflow-y-auto p-6">
      {#if error}
        <div class="mb-4 px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
          <button onclick={() => error = null} class="ml-2 text-red-400 hover:text-red-600">&times;</button>
        </div>
      {/if}

      {#if !datasetUploaded}
        <div class="flex items-center justify-center h-full">
          <div class="text-center">
            <div class="text-6xl mb-4 opacity-30">&#128202;</div>
            <h2 class="text-xl font-semibold text-gray-700 mb-2">Upload Dataset to Begin</h2>
            <p class="text-gray-400 mb-6">Upload a CSV file with your classification dataset</p>
            <label class="inline-flex items-center gap-2 px-6 py-3 bg-[var(--navy)] text-white rounded-lg font-medium cursor-pointer hover:bg-[#2a3580] transition-colors shadow-md">
              Choose CSV File
              <input type="file" accept=".csv" onchange={handleUpload} class="hidden" />
            </label>
          </div>
        </div>
      {:else}
        <div class="flex items-center justify-between mb-6">
          <div>
            <h2 class="text-2xl font-bold text-[var(--navy)]">{stageTitles[activeStage]}</h2>
            <div class="mt-1"><StatusBadge status={stageStatus[activeStage]} /></div>
          </div>
          {#if canRunActive()}
            <button
              onclick={handleRunActive}
              class="px-6 py-2.5 bg-[var(--accent)] text-[var(--navy)] font-semibold rounded-lg hover:bg-[#6bb8d3] transition-colors shadow-md"
            >
              Run Stage
            </button>
          {/if}
        </div>

        {#if running}
          <div class="flex flex-col items-center justify-center py-20">
            <div class="w-12 h-12 border-4 border-[var(--ice-blue)] border-t-[var(--navy)] rounded-full animate-spin"></div>
            <p class="mt-4 text-gray-500">Computing...</p>
          </div>
        {:else if activeStage === "etl"}
          <ETLStage
            status={stageStatus.etl}
            onconfirmed={handleConfirmed}
          />
        {:else if activeStage === "stats"}
          <StatsStage
            data={stageData.stats}
            status={stageStatus.stats}
            onconfirmed={handleConfirmed}
          />
        {:else if activeStage === "model"}
          <ModelStage
            data={stageData.model}
            status={stageStatus.model}
            onconfirmed={handleConfirmed}
            onrun={handleRunModel}
            canRun={stageStatus.stats === "confirmed" || stageStatus.stats === "complete"}
          />
        {:else if activeStage === "evaluate"}
          <EvaluateStage
            data={stageData.evaluate}
            status={stageStatus.evaluate}
            onconfirmed={handleConfirmed}
          />
        {/if}
      {/if}
    </main>
  </div>
</div>
