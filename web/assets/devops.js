const $ = id => document.getElementById(id);
const view = { tasks: [], job: null, selected: null, health: null, polling: false, detailRequest: 0, sourceDetail: null, workload: 'project' };
const labels = { query: '查询空白处理', offset: '分页偏移', limit: '结果数量边界', active: '状态参数解析', unicode: 'Unicode 匹配', tenant_cache: '租户隔离与缓存串租', cursor: '重复排序键与游标绑定', invalidation: '写后缓存与版本失效', policy: '动态配置与租户策略', transactions: '原子更新、幂等与重启' };
const statusLabels = { queued: '排队中', running: '执行中', verified: '验收通过', blocked: '依赖阻塞', budget_exhausted: '预算耗尽', error: '执行错误', failed: '验收失败', complete: '序列完成' };
const pretty = value => JSON.stringify(value, null, 2);

async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

async function loadTasks(workload) {
  const data = await api(`/api/devops/tasks?workload=${encodeURIComponent(workload)}`);
  view.tasks = data.tasks; view.workload = workload; $('workload').value = workload;
}

function renderSources() {
  const detail = view.sourceDetail;
  if (!detail) return;
  const file = $('source-file').value;
  $('candidate-source').textContent = detail.candidate_files ? detail.candidate_files[file] ?? '该文件不存在' : detail.candidate;
  $('deployed-source').textContent = detail.deployed_files ? detail.deployed_files[file] ?? '该文件不存在' : detail.deployed;
}

function setBusy(busy) {
  $('experiment-form').querySelectorAll('input,select,button').forEach(el => { el.disabled = busy; });
  $('start-experiment').textContent = busy ? '正在执行关联序列…' : '运行十轮闭环';
}

function renderTasks() {
  const report = view.job?.report;
  const rows = new Map((report?.tasks || []).map(row => [row.task_id, row]));
  $('sequence-tasks').replaceChildren(...view.tasks.map(task => {
    const row = rows.get(task.task_id);
    const current = row ? (row.passed ? 'verified' : row.status === 'verified' ? 'failed' : row.status)
      : report?.active_task === task.task_id ? 'running' : 'queued';
    const button = document.createElement('button');
    button.type = 'button'; button.className = `sequence-task ${current}${view.selected === task.task_id ? ' selected' : ''}`;
    button.disabled = !row; button.setAttribute('aria-pressed', String(view.selected === task.task_id));
    const title = document.createElement('strong'); title.textContent = `${task.task_id} / ${task.group === 'ops' ? '运维' : '研发'}`;
    const label = document.createElement('span'); label.textContent = labels[task.incident] || task.incident;
    const status = document.createElement('small'); status.textContent = row ? statusLabels[current] || current : current === 'running' ? '执行中' : '等待依赖';
    button.append(title, label, status); button.addEventListener('click', () => inspect(task.task_id));
    return button;
  }));
  $('sequence-progress').value = rows.size;
}

function renderJob(job) {
  view.job = job;
  const report = job.report;
  const rows = report?.tasks || [];
  const sum = key => rows.reduce((total, row) => total + (row[key] || 0), 0);
  $('passed-total').textContent = `${rows.filter(r => r.passed).length} / 10`;
  $('reuse-total').textContent = sum('effective_cross_group_reuse');
  $('transfer-total').textContent = `${sum('total_transfer_bytes').toLocaleString()} B`;
  $('calls-total').textContent = sum('llm_calls');
  $('run-status').textContent = `${statusLabels[job.status] || job.status} · ${job.workload || 'service'} · ${job.backend} · ${job.mode}${report?.active_task ? ` · 当前 ${report.active_task}` : ''}`;
  if (job.error) $('experiment-error').textContent = job.error;
  renderTasks();
  if (!view.selected && rows.length) inspect(rows[0].task_id);
}

async function inspect(taskId) {
  const request = ++view.detailRequest;
  view.selected = taskId; renderTasks();
  view.sourceDetail = null; $('source-picker').hidden = true;
  const row = view.job.report.tasks.find(r => r.task_id === taskId);
  const task = view.tasks.find(t => t.task_id === taskId);
  $('inspector-heading').textContent = `${taskId} · ${task.group === 'ops' ? '运维处置' : '研发修复'}`;
  $('task-outcome').textContent = row.passed ? 'PASS' : row.status === 'blocked' ? 'BLOCKED' : 'FAIL';
  $('task-summary').textContent = row.summary || row.error || row.reason || task.instruction;
  $('task-evidence').textContent = pretty({ observations: row.observations || [], grade: row.grade || null });
  $('task-memory').textContent = pretty({ used: row.used_memory_ids || [], rejected: row.rejected_memory_ids || [], cross_group_reuse: row.cross_group_reuse || 0, runtime_context_filtered: row.context_filtered_reuse || 0 });
  const facts = [['步骤', row.steps ?? '—'], ['模型调用', row.llm_calls ?? 0], ['任务耗时', `${(row.elapsed_sec || 0).toFixed(3)}s`], ['状态载荷', `${row.state_payload_bytes || 0} B`], ['修改文件数', row.patch_files || 0], ['场景业务操作数', row.probe_operations || 0]];
  $('task-facts').replaceChildren(...facts.map(([key, value]) => {
    const block = document.createElement('div'); const dt = document.createElement('dt'); const dd = document.createElement('dd');
    dt.textContent = key; dd.textContent = value; block.append(dt, dd); return block;
  }));
  for (const id of ['candidate-source', 'deployed-source', 'task-trace']) $(id).textContent = row.status === 'blocked' ? '依赖阻塞，没有执行产物。' : '读取运行产物…';
  if (row.status === 'blocked') return;
  $('inspector').setAttribute('aria-busy', 'true');
  try {
    const detail = await api(`/api/devops/runs/${view.job.job_id}/tasks/${taskId}`);
    if (request !== view.detailRequest) return;
    view.sourceDetail = detail;
    if (detail.candidate_files) {
      $('source-file').replaceChildren(...Object.keys(detail.candidate_files).map(name => {
        const option = document.createElement('option'); option.value = name; option.textContent = name; return option;
      }));
      const changed = (row.observations || []).flatMap(o => o.changed_files || []);
      if (changed.length) $('source-file').value = changed[0];
      $('source-picker').hidden = false;
    }
    renderSources();
    $('task-trace').textContent = detail.trace.map(frame => {
      try { return pretty(JSON.parse(frame)); } catch { return frame; }
    }).join('\n\n');
  } catch (error) {
    if (request === view.detailRequest) $('task-trace').textContent = `产物读取失败：${error.message}`;
  } finally { if (request === view.detailRequest) $('inspector').setAttribute('aria-busy', 'false'); }
}

async function watch(jobId) {
  if (view.polling) return;
  view.polling = true; setBusy(true); $('reconnect').hidden = true;
  try {
    for (;;) {
      const job = await api(`/api/devops/runs/${jobId}`); renderJob(job);
      if (!['queued', 'running'].includes(job.status)) break;
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  } catch (error) {
    $('experiment-error').textContent = `连接中断：${error.message}。后台任务可能仍在执行。`;
    $('reconnect').hidden = false;
  } finally { view.polling = false; setBusy(false); }
}

async function start(event) {
  event.preventDefault(); $('experiment-error').textContent = ''; setBusy(true);
  try {
    const previous = sessionStorage.getItem('devops-job');
    if (previous) {
      const job = await api(`/api/devops/runs/${previous}`).catch(() => null);
      if (job && ['queued', 'running'].includes(job.status)) { await loadTasks(job.workload || 'service'); await watch(previous); return; }
    }
    const backend = $('backend').value;
    view.health = await api('/api/health');
    if (backend !== view.health.mode || $('generation-device').value !== view.health.device || $('embedding-device').value !== view.health.embedding_device) {
      $('run-status').textContent = '准备引擎；Qwen 首次加载可能需要较长时间…';
      view.health = await api('/api/engine', { backend, device: $('generation-device').value, embedding_device: $('embedding-device').value });
      if (view.health.last_error) throw new Error(view.health.last_error);
      $('engine-status').textContent = `${view.health.mode} engine ready`;
    }
    await loadTasks($('workload').value);
    const job = await api('/api/devops/runs', { mode: $('experiment-mode').value, max_steps: Number($('max-steps').value), workload: view.workload });
    sessionStorage.setItem('devops-job', job.job_id);
    ++view.detailRequest; view.selected = null; renderJob(job); await watch(job.job_id);
  } catch (error) { $('experiment-error').textContent = error.message; }
  finally { setBusy(false); }
}

async function boot() {
  const [health] = await Promise.all([api('/api/health'), loadTasks($('workload').value)]);
  view.health = health;
  $('backend').value = health.mode; $('generation-device').value = health.device; $('embedding-device').value = health.embedding_device;
  $('engine-status').textContent = `${health.mode} engine ready`;
  renderTasks(); $('experiment-form').addEventListener('submit', start);
  $('reconnect').addEventListener('click', () => watch(sessionStorage.getItem('devops-job')));
  $('source-file').addEventListener('change', renderSources);
  $('workload').addEventListener('change', async () => {
    setBusy(true);
    try {
      await loadTasks($('workload').value); ++view.detailRequest; view.selected = null;
      view.sourceDetail = null; $('source-picker').hidden = true;
      renderJob({ status: 'queued', backend: view.health.mode, mode: $('experiment-mode').value, workload: view.workload, report: null });
      $('inspector-heading').textContent = '选择一轮任务'; $('task-summary').textContent = '运行新环境后查看执行证据。';
      for (const id of ['task-evidence', 'task-memory', 'candidate-source', 'deployed-source', 'task-trace']) $(id).textContent = '等待新任务运行…';
      $('task-facts').replaceChildren(); $('task-outcome').textContent = 'EMPTY';
    } catch (error) { $('experiment-error').textContent = error.message; }
    finally { setBusy(false); }
  });
  setBusy(false);
  const previous = sessionStorage.getItem('devops-job');
  if (previous) {
    const job = await api(`/api/devops/runs/${previous}`).catch(() => null);
    if (job) { await loadTasks(job.workload || 'service'); renderJob(job); if (['queued', 'running'].includes(job.status)) await watch(previous); }
    else sessionStorage.removeItem('devops-job');
  }
}
boot().catch(error => { $('experiment-error').textContent = error.message; $('engine-status').textContent = 'API unavailable'; $('start-experiment').disabled = true; });
