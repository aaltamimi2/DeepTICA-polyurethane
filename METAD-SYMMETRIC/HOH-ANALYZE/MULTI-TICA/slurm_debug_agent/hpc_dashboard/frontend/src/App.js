import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Server,
  Activity,
  Clock,
  Cpu,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader2,
  RefreshCw,
  Timer,
  Layers,
  Zap,
  Terminal,
  ChevronDown,
  ChevronUp,
  X,
  Filter,
  FileText,
  Settings,
  GitBranch,
  Play,
  Edit3,
  MessageCircle,
  Send,
  ArrowUpDown,
  CheckSquare,
  Square,
  Users,
  BarChart3,
  Sparkles,
  StickyNote,
  Plus,
  Trash2
} from 'lucide-react';
import './index.css';

// ============================================================
// Configuration
// ============================================================

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8081/ws';
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8081/api';

// ============================================================
// Utility Functions
// ============================================================

function formatTime(timeStr) {
  if (!timeStr) return '--:--:--';
  return timeStr;
}

function getStateColor(state) {
  const colors = {
    RUNNING: 'running',
    PENDING: 'pending',
    COMPLETED: 'completed',
    FAILED: 'failed',
    CANCELLED: 'failed',
    TIMEOUT: 'timeout',
    NODE_FAIL: 'failed'
  };
  return colors[state] || 'pending';
}

function getStateIcon(state) {
  switch (state) {
    case 'RUNNING':
      return <Loader2 size={14} className="animate-spin" />;
    case 'PENDING':
      return <Clock size={14} />;
    case 'COMPLETED':
      return <CheckCircle size={14} />;
    case 'FAILED':
    case 'CANCELLED':
    case 'NODE_FAIL':
      return <XCircle size={14} />;
    case 'TIMEOUT':
      return <Timer size={14} />;
    default:
      return <Activity size={14} />;
  }
}

function getStateCategory(state) {
  if (state === 'RUNNING') return 'running';
  if (state === 'PENDING') return 'pending';
  if (state === 'COMPLETED') return 'completed';
  return 'failed'; // FAILED, TIMEOUT, CANCELLED, NODE_FAIL
}

// ============================================================
// Components
// ============================================================

// Connection Status Indicator
function ConnectionStatus({ status }) {
  const statusConfig = {
    connected: { dot: 'connected', text: 'Connected', color: 'var(--success)' },
    disconnected: { dot: 'disconnected', text: 'Disconnected', color: 'var(--error)' },
    connecting: { dot: 'connecting', text: 'Connecting...', color: 'var(--warning)' }
  };

  const config = statusConfig[status] || statusConfig.disconnected;

  return (
    <div className="connection-status">
      <div className={`connection-dot ${config.dot}`} />
      <span style={{ color: config.color }}>{config.text}</span>
    </div>
  );
}

// Stats Summary Bar with clickable filters
function StatsBar({ jobs, statusFilter, onStatusFilterChange }) {
  const stats = {
    running: jobs.filter(j => j.state === 'RUNNING').length,
    pending: jobs.filter(j => j.state === 'PENDING').length,
    completed: jobs.filter(j => j.state === 'COMPLETED').length,
    failed: jobs.filter(j => ['FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL'].includes(j.state)).length,
    total: jobs.length
  };

  const filterItems = [
    { key: 'running', icon: Loader2, color: 'var(--running)', label: 'Running', count: stats.running, spin: true },
    { key: 'pending', icon: Clock, color: 'var(--pending)', label: 'Pending', count: stats.pending },
    { key: 'completed', icon: CheckCircle, color: 'var(--success)', label: 'Completed', count: stats.completed },
    { key: 'failed', icon: XCircle, color: 'var(--error)', label: 'Failed', count: stats.failed },
  ];

  return (
    <div className="stats-bar">
      {filterItems.map(item => {
        const Icon = item.icon;
        const isActive = statusFilter === item.key;
        return (
          <button
            key={item.key}
            onClick={() => onStatusFilterChange(isActive ? null : item.key)}
            className={`stat-item ${isActive ? 'active' : ''}`}
            style={{
              cursor: 'pointer',
              background: isActive ? 'var(--bg-tertiary)' : 'transparent',
              border: isActive ? '2px solid var(--primary)' : '2px solid transparent',
              borderRadius: '8px',
              padding: '12px 16px',
            }}
          >
            <Icon size={18} style={{ color: item.color }} className={item.spin ? 'animate-spin' : ''} />
            <div>
              <div className="stat-value">{item.count}</div>
              <div className="stat-label">{item.label}</div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

// Cluster Filter
function ClusterFilter({ clusters, clusterFilter, onClusterFilterChange }) {
  const clusterList = Object.entries(clusters);

  if (clusterList.length <= 1) return null;

  return (
    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
      <Filter size={14} style={{ color: 'var(--text-tertiary)' }} />
      <select
        value={clusterFilter || 'all'}
        onChange={(e) => onClusterFilterChange(e.target.value === 'all' ? null : e.target.value)}
        style={{
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border-color)',
          borderRadius: '6px',
          padding: '6px 12px',
          color: 'var(--text-primary)',
          fontSize: '13px',
          cursor: 'pointer'
        }}
      >
        <option value="all">All Clusters</option>
        {clusterList.map(([id, cluster]) => (
          <option key={id} value={cluster.name}>{cluster.name}</option>
        ))}
      </select>
    </div>
  );
}

// Extract project folder from work directory path
function extractProjectFolder(workDir) {
  if (!workDir) return 'Unknown';
  // Extract the last meaningful folder from paths like:
  // /home/aaltamimi/SURFACTANT-PIPELINE
  // /expanse/lustre/scratch/aaltamimi/temp_project/SOME-PROJECT
  // /srv/home/aaltamimi2/PROJECT
  const parts = workDir.split('/').filter(p => p);
  // Skip common prefixes
  const skipFolders = ['home', 'srv', 'expanse', 'lustre', 'scratch', 'temp_project'];
  for (let i = parts.length - 1; i >= 0; i--) {
    const part = parts[i];
    if (!skipFolders.includes(part.toLowerCase()) && !part.match(/^[a-z]+\d*$/i)) {
      return part;
    }
  }
  // Fallback to last part
  return parts[parts.length - 1] || 'Unknown';
}

// Work Directory Filter
function WorkDirFilter({ jobs, workDirFilter, onWorkDirFilterChange }) {
  // Extract unique project folders
  const projectFolders = [...new Set(jobs.map(j => extractProjectFolder(j.work_dir)))].sort();

  if (projectFolders.length <= 1) return null;

  return (
    <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
      <select
        value={workDirFilter || 'all'}
        onChange={(e) => onWorkDirFilterChange(e.target.value === 'all' ? null : e.target.value)}
        style={{
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border-color)',
          borderRadius: '6px',
          padding: '6px 12px',
          color: 'var(--text-primary)',
          fontSize: '13px',
          cursor: 'pointer',
          maxWidth: '200px'
        }}
      >
        <option value="all">All Projects</option>
        {projectFolders.map(folder => (
          <option key={folder} value={folder}>{folder}</option>
        ))}
      </select>
    </div>
  );
}

// Progress Bar Component
function ProgressBar({ progress, state }) {
  const color = state === 'RUNNING' ? 'var(--primary)' :
                state === 'COMPLETED' ? 'var(--success)' :
                state === 'FAILED' ? 'var(--error)' : 'var(--text-tertiary)';

  return (
    <div className="progress-bar" style={{ marginTop: '8px' }}>
      <div
        className="progress-bar-fill"
        style={{
          width: `${progress}%`,
          background: color
        }}
      />
    </div>
  );
}

// Diagnosis Log Panel
function DiagnosisLog({ diagnosisLog, onClose }) {
  if (!diagnosisLog || diagnosisLog.length === 0) {
    return (
      <div style={{
        position: 'fixed',
        right: 0,
        top: 0,
        bottom: 0,
        width: '400px',
        background: 'var(--bg-secondary)',
        borderLeft: '1px solid var(--border-color)',
        padding: '16px',
        overflowY: 'auto',
        zIndex: 1000
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <FileText size={18} />
            Diagnosis Log
          </h3>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}>
            <X size={18} />
          </button>
        </div>
        <p style={{ color: 'var(--text-tertiary)', fontSize: '13px' }}>No diagnoses yet. Run a diagnosis on a failed job to see results here.</p>
      </div>
    );
  }

  return (
    <div style={{
      position: 'fixed',
      right: 0,
      top: 0,
      bottom: 0,
      width: '450px',
      background: 'var(--bg-secondary)',
      borderLeft: '1px solid var(--border-color)',
      padding: '16px',
      overflowY: 'auto',
      zIndex: 1000
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileText size={18} />
          Diagnosis Log ({diagnosisLog.length})
        </h3>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}>
          <X size={18} />
        </button>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {diagnosisLog.map((entry, i) => (
          <div key={i} style={{
            padding: '12px',
            background: 'var(--bg-tertiary)',
            borderRadius: '8px',
            borderLeft: '3px solid var(--primary)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span className="mono" style={{ fontWeight: '600' }}>Job {entry.job_id}</span>
              <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>{entry.timestamp}</span>
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap', maxHeight: '200px', overflowY: 'auto' }}>
              {entry.diagnosis}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Analytics Panel (GPU/Walltime Usage)
function AnalyticsPanel({ onClose }) {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [goalInput, setGoalInput] = useState('');
  const [editingGoal, setEditingGoal] = useState(false);

  const fetchAnalytics = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/analytics`);
      const data = await response.json();
      setAnalytics(data);
      setGoalInput(data.monthly_goal.toString());
    } catch (e) {
      console.error('Failed to fetch analytics:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [fetchAnalytics]);

  const handleSetGoal = async () => {
    const goal = parseFloat(goalInput);
    if (isNaN(goal) || goal <= 0) return;

    try {
      await fetch(`${API_URL}/analytics/goal?goal_hours=${goal}`, { method: 'POST' });
      setEditingGoal(false);
      fetchAnalytics();
    } catch (e) {
      console.error('Failed to set goal:', e);
    }
  };

  const formatHours = (hours) => {
    if (hours >= 24) {
      const days = Math.floor(hours / 24);
      const remainingHours = hours % 24;
      return `${days}d ${remainingHours.toFixed(1)}h`;
    }
    return `${hours.toFixed(1)}h`;
  };

  return (
    <div style={{
      position: 'fixed',
      right: 0,
      top: 0,
      bottom: 0,
      width: '400px',
      background: 'var(--bg-secondary)',
      borderLeft: '1px solid var(--border-color)',
      padding: '16px',
      zIndex: 1000,
      overflowY: 'auto'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity size={18} style={{ color: 'var(--primary)' }} />
          Usage Analytics
        </h3>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}>
          <X size={18} />
        </button>
      </div>

      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '40px' }}>
          <Loader2 className="animate-spin" size={24} />
        </div>
      ) : analytics && (
        <>
          {/* Monthly Goal Progress */}
          <div style={{
            background: 'var(--bg-tertiary)',
            borderRadius: '12px',
            padding: '16px',
            marginBottom: '20px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>Monthly Usage Goal</span>
              <button
                onClick={() => setEditingGoal(!editingGoal)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--primary)',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                {editingGoal ? 'Cancel' : 'Edit'}
              </button>
            </div>

            {editingGoal ? (
              <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
                <input
                  type="number"
                  value={goalInput}
                  onChange={(e) => setGoalInput(e.target.value)}
                  style={{
                    flex: 1,
                    padding: '8px 12px',
                    background: 'var(--bg-secondary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    color: 'var(--text-primary)',
                    fontSize: '14px'
                  }}
                  placeholder="Hours per month"
                />
                <button
                  onClick={handleSetGoal}
                  style={{
                    padding: '8px 16px',
                    background: 'var(--primary)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer'
                  }}
                >
                  Save
                </button>
              </div>
            ) : null}

            <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginBottom: '8px' }}>
              <span style={{ fontSize: '28px', fontWeight: '600', color: 'var(--text-primary)' }}>
                {formatHours(analytics.total_hours)}
              </span>
              <span style={{ fontSize: '14px', color: 'var(--text-tertiary)' }}>
                / {formatHours(analytics.monthly_goal)}
              </span>
            </div>

            {/* Progress Bar */}
            <div style={{
              height: '8px',
              background: 'var(--bg-secondary)',
              borderRadius: '4px',
              overflow: 'hidden',
              marginBottom: '8px'
            }}>
              <div style={{
                height: '100%',
                width: `${Math.min(analytics.usage_percent, 100)}%`,
                background: analytics.usage_percent > 90 ? 'var(--error)' :
                           analytics.usage_percent > 70 ? 'var(--warning)' : 'var(--success)',
                borderRadius: '4px',
                transition: 'width 0.3s ease'
              }} />
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', textAlign: 'right' }}>
              {analytics.usage_percent}% of monthly goal
            </div>
          </div>

          {/* Summary Stats */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '12px',
            marginBottom: '20px'
          }}>
            <div style={{
              background: 'var(--running-bg)',
              borderRadius: '8px',
              padding: '12px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '24px', fontWeight: '600', color: 'var(--running)' }}>
                {formatHours(analytics.running_hours)}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>Currently Running</div>
            </div>
            <div style={{
              background: 'var(--bg-tertiary)',
              borderRadius: '8px',
              padding: '12px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '24px', fontWeight: '600', color: 'var(--text-primary)' }}>
                {analytics.total_jobs}
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>Total Jobs</div>
            </div>
          </div>

          {/* Per-Cluster Stats */}
          <h4 style={{ marginBottom: '12px', fontSize: '14px', color: 'var(--text-secondary)' }}>
            Per-Cluster Usage
          </h4>
          {Object.entries(analytics.cluster_stats).map(([cluster, stats]) => (
            <div key={cluster} style={{
              background: 'var(--bg-tertiary)',
              borderRadius: '8px',
              padding: '12px',
              marginBottom: '8px'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ fontWeight: '500', fontSize: '13px' }}>{cluster}</span>
                <span style={{ fontSize: '16px', fontWeight: '600', color: 'var(--primary)' }}>
                  {formatHours(stats.total_hours)}
                </span>
              </div>
              <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: 'var(--text-tertiary)' }}>
                <span style={{ color: 'var(--running)' }}>{stats.running_jobs} running</span>
                <span style={{ color: 'var(--success)' }}>{stats.completed_jobs} completed</span>
                <span style={{ color: 'var(--error)' }}>{stats.failed_jobs} failed</span>
              </div>
            </div>
          ))}

          {/* Refresh button */}
          <button
            onClick={fetchAnalytics}
            style={{
              width: '100%',
              padding: '10px',
              background: 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              marginTop: '16px'
            }}
          >
            <RefreshCw size={14} />
            Refresh Analytics
          </button>
        </>
      )}
    </div>
  );
}

// Troubleshoot Modal with Auto-Fix and Preview
// Code Diff Component
function CodeDiff({ original, proposed, filename, type }) {
  const [expanded, setExpanded] = useState(true);

  return (
    <div style={{
      marginBottom: '12px',
      border: '1px solid var(--border-color)',
      borderRadius: '8px',
      overflow: 'hidden'
    }}>
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          background: 'var(--bg-tertiary)',
          cursor: 'pointer',
          borderBottom: expanded ? '1px solid var(--border-color)' : 'none'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileText size={14} style={{ color: type === 'script' ? 'var(--warning)' : 'var(--primary)' }} />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', fontWeight: '500' }}>
            {filename}
          </span>
          <span style={{
            padding: '2px 6px',
            background: original ? 'var(--warning)' : 'var(--success)',
            color: 'white',
            borderRadius: '3px',
            fontSize: '10px'
          }}>
            {original ? 'MODIFIED' : 'NEW'}
          </span>
        </div>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </div>
      {expanded && (
        <div style={{ maxHeight: '300px', overflow: 'auto' }}>
          <pre style={{
            margin: 0,
            padding: '12px',
            background: 'var(--bg-primary)',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            lineHeight: '1.5',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all'
          }}>
            {proposed.split('\n').map((line, i) => {
              const isComment = line.trim().startsWith('#');
              const isModification = line.includes('=') && !isComment;
              return (
                <div
                  key={i}
                  style={{
                    background: isModification ? 'rgba(34, 197, 94, 0.1)' : 'transparent',
                    borderLeft: isModification ? '3px solid var(--success)' : '3px solid transparent',
                    paddingLeft: '8px',
                    marginLeft: '-12px',
                    color: isComment ? 'var(--text-tertiary)' : isModification ? 'var(--success)' : 'var(--text-primary)'
                  }}
                >
                  <span style={{ color: 'var(--text-tertiary)', marginRight: '12px', userSelect: 'none' }}>
                    {String(i + 1).padStart(3, ' ')}
                  </span>
                  {line || ' '}
                </div>
              );
            })}
          </pre>
        </div>
      )}
    </div>
  );
}

function TroubleshootModal({ job, onClose, onSubmit }) {
  const [modifications, setModifications] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [continueFrom, setContinueFrom] = useState('start');
  const [preview, setPreview] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [diagnosisEdits, setDiagnosisEdits] = useState([]);
  const [loadingEdits, setLoadingEdits] = useState(false);
  const [pipelineFlags, setPipelineFlags] = useState([]);
  const [loadingFlags, setLoadingFlags] = useState(false);
  const [fullPreview, setFullPreview] = useState(null);
  const [showCodeReview, setShowCodeReview] = useState(false);
  const [loadingFullPreview, setLoadingFullPreview] = useState(false);
  const previewTimeoutRef = useRef(null);

  // Fetch diagnosis edits on mount
  useEffect(() => {
    if (job?.job_id) {
      setLoadingEdits(true);
      fetch(`${API_URL}/jobs/${job.job_id}/diagnosis/edits`)
        .then(res => res.json())
        .then(data => {
          if (data.edits && data.edits.length > 0) {
            setDiagnosisEdits(data.edits);
          }
        })
        .catch(err => console.error('Failed to fetch diagnosis edits:', err))
        .finally(() => setLoadingEdits(false));
    }
  }, [job?.job_id]);

  // Fetch dynamic pipeline flags on mount
  useEffect(() => {
    if (job?.job_id) {
      setLoadingFlags(true);
      fetch(`${API_URL}/jobs/${job.job_id}/pipeline-flags`)
        .then(res => res.json())
        .then(data => {
          if (data.flags && data.flags.length > 0) {
            setPipelineFlags(data.flags);
          }
        })
        .catch(err => console.error('Failed to fetch pipeline flags:', err))
        .finally(() => setLoadingFlags(false));
    }
  }, [job?.job_id]);

  // Debounced preview fetch
  useEffect(() => {
    if (!modifications.trim()) {
      setPreview(null);
      return;
    }

    if (previewTimeoutRef.current) {
      clearTimeout(previewTimeoutRef.current);
    }

    previewTimeoutRef.current = setTimeout(async () => {
      setPreviewLoading(true);
      try {
        const response = await fetch(`${API_URL}/jobs/${job.job_id}/troubleshoot/preview`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ modifications, cluster: job.cluster })
        });
        const data = await response.json();
        setPreview(data);
      } catch (e) {
        console.error('Preview failed:', e);
      }
      setPreviewLoading(false);
    }, 500);

    return () => {
      if (previewTimeoutRef.current) {
        clearTimeout(previewTimeoutRef.current);
      }
    };
  }, [modifications, job?.job_id, job?.cluster]);

  if (!job) return null;

  // Default pipeline steps (used as fallback if no flags fetched)
  const defaultPipelineSteps = [
    { id: 'start', label: 'Start Fresh', description: 'Run entire pipeline from beginning', skipFlags: [] },
    { id: 'build', label: 'After SwarmCG', description: 'Skip AA/CG parameter prep (SwarmCG)', skipFlags: ['--skip-swarmcg'] },
    { id: 'equil', label: 'After Build', description: 'Skip SwarmCG and system building', skipFlags: ['--skip-swarmcg', '--skip-build'] },
    { id: 'metad', label: 'After Equilibration', description: 'Skip to metadynamics only', skipFlags: ['--skip-swarmcg', '--skip-build', '--skip-equil'] },
  ];

  // Use dynamic flags if available, otherwise use defaults
  const pipelineSteps = pipelineFlags.length > 0
    ? [
        { id: 'start', label: 'Start Fresh', description: 'Run entire pipeline from beginning', skipFlags: [] },
        ...pipelineFlags.map((f, i) => ({
          id: f.stage,
          label: `After ${f.stage.charAt(0).toUpperCase() + f.stage.slice(1).replace(/[-_]/g, ' ')}`,
          description: f.description || `Skip ${f.stage} stage`,
          skipFlags: pipelineFlags.slice(0, i + 1).map(pf => pf.flag)
        }))
      ]
    : defaultPipelineSteps;

  // Extract recommendations from diagnosis
  const getRecommendations = () => {
    if (!job.diagnosis) return [];
    const lines = job.diagnosis.split('\n');
    const recs = [];
    let inRecs = false;
    for (const line of lines) {
      if (line.includes('Recommendation') || line.includes('resolve') || line.includes('consider')) {
        inRecs = true;
      }
      if (inRecs && line.trim().startsWith('*')) {
        recs.push(line.trim().replace(/^\*\s*/, '').replace(/\*\*/g, ''));
      }
    }
    return recs.slice(0, 5);
  };

  // Suggest continue point based on diagnosis
  const getSuggestedContinuePoint = () => {
    if (!job.diagnosis) return null;
    const diag = job.diagnosis.toLowerCase();
    if (diag.includes('metadynamics') && (diag.includes('failed') || diag.includes('error'))) {
      return 'metad';
    }
    if (diag.includes('equilibrat') && (diag.includes('failed') || diag.includes('error'))) {
      return 'equil';
    }
    if (diag.includes('insert') || diag.includes('build') || diag.includes('system assembly')) {
      return 'build';
    }
    if (diag.includes('swarmcg') || diag.includes('acpype') || diag.includes('parameter')) {
      return 'start';
    }
    return null;
  };

  // Apply AI recommendations as modifications
  const applyAIFix = () => {
    const fixTexts = diagnosisEdits.map(edit => {
      const action = edit.action === 'set' ? 'Set' :
                    edit.action === 'reduce' ? 'Reduce' :
                    edit.action === 'increase' ? 'Increase' : edit.action;
      return `${action} ${edit.param.replace(/_/g, ' ').toLowerCase()}${edit.value ? ` to ${edit.value}` : ''}`;
    });

    if (fixTexts.length > 0) {
      setModifications(fixTexts.join(', '));
    }
  };

  const suggestedPoint = getSuggestedContinuePoint();

  // Fetch full preview with code diffs
  const handleReviewCode = async () => {
    setLoadingFullPreview(true);
    const selectedStep = pipelineSteps.find(s => s.id === continueFrom);
    try {
      const response = await fetch(`${API_URL}/jobs/${job.job_id}/troubleshoot/full-preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modifications: modifications,
          cluster: job.cluster,
          skip_flags: selectedStep ? selectedStep.skipFlags : []
        })
      });
      const data = await response.json();
      setFullPreview(data);
      setShowCodeReview(true);
    } catch (e) {
      console.error('Full preview failed:', e);
    }
    setLoadingFullPreview(false);
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    const selectedStep = pipelineSteps.find(s => s.id === continueFrom);
    try {
      const response = await fetch(`${API_URL}/jobs/${job.job_id}/troubleshoot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: job.job_id,
          modifications: modifications,
          work_dir: job.work_dir,
          job_name: job.name,
          skip_flags: selectedStep ? selectedStep.skipFlags : []
        })
      });
      const data = await response.json();
      setResult(data);
      setShowCodeReview(false);
    } catch (e) {
      setResult({ status: 'error', message: e.message });
    }
    setSubmitting(false);
  };

  const recommendations = getRecommendations();

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0, 0, 0, 0.6)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 2000,
        animation: 'fadeIn 0.15s ease-out'
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--bg-secondary)',
          borderRadius: '12px',
          maxWidth: '600px',
          maxHeight: '85vh',
          width: '90%',
          overflow: 'hidden',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
          animation: 'slideUp 0.2s ease-out'
        }}
      >
        <div style={{
          padding: '16px 20px',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'var(--bg-tertiary)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <GitBranch size={18} style={{ color: 'var(--warning)' }} />
            <span style={{ fontWeight: '600', fontSize: '16px' }}>Troubleshoot Job</span>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              borderRadius: '6px',
              padding: '8px',
              cursor: 'pointer',
              color: 'var(--text-secondary)'
            }}
          >
            <X size={18} />
          </button>
        </div>

        <div style={{ padding: '20px', overflowY: 'auto', maxHeight: 'calc(85vh - 140px)' }}>
          {result ? (
            <div style={{
              padding: '16px',
              background: result.status === 'success' ? 'var(--success-bg)' : 'var(--error-bg)',
              borderRadius: '8px',
              marginBottom: '16px'
            }}>
              <div style={{
                fontWeight: '600',
                color: result.status === 'success' ? 'var(--success)' : 'var(--error)',
                marginBottom: '8px'
              }}>
                {result.status === 'success' ? 'Troubleshooting Job Created!' : 'Error'}
              </div>
              <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                {result.message}
              </div>
              {result.new_job_id && (
                <div style={{ marginTop: '8px', fontFamily: 'var(--font-mono)', fontSize: '13px' }}>
                  New Job ID: <strong>{result.new_job_id}</strong>
                </div>
              )}
              {result.troubleshoot_attempt && (
                <div style={{ marginTop: '4px', fontSize: '12px', color: 'var(--text-secondary)' }}>
                  <GitBranch size={12} style={{ display: 'inline', marginRight: '4px' }} />
                  Troubleshoot attempt #{result.troubleshoot_attempt} for job {result.parent_job_id}
                </div>
              )}
              {result.troubleshoot_dir && (
                <div style={{ marginTop: '4px', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-tertiary)' }}>
                  Dir: {result.troubleshoot_dir}
                </div>
              )}
              {/* Show applied config changes */}
              {result.parsed_changes && result.parsed_changes.length > 0 && (
                <div style={{ marginTop: '12px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginBottom: '6px', textTransform: 'uppercase' }}>
                    Applied Configuration Changes
                  </div>
                  {result.parsed_changes.map((change, i) => (
                    <div key={i} style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      marginBottom: '4px',
                      padding: '4px 8px',
                      background: 'rgba(255,255,255,0.1)',
                      borderRadius: '4px',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '11px'
                    }}>
                      <span style={{ color: 'var(--success)' }}>{change.param}</span>
                      <span style={{ color: 'var(--text-tertiary)' }}>→</span>
                      <span>{change.value}</span>
                    </div>
                  ))}
                </div>
              )}
              {/* Show SLURM config used */}
              {result.slurm_config && (
                <div style={{ marginTop: '8px', fontSize: '11px', color: 'var(--text-tertiary)' }}>
                  Cluster: {result.slurm_config.partition} | Account: {result.slurm_config.account}
                </div>
              )}
            </div>
          ) : showCodeReview && fullPreview ? (
            /* Code Review Mode - Show file diffs */
            <div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                marginBottom: '16px',
                padding: '12px',
                background: 'var(--bg-tertiary)',
                borderRadius: '8px'
              }}>
                <Terminal size={18} style={{ color: 'var(--primary)' }} />
                <div>
                  <div style={{ fontWeight: '600', fontSize: '14px' }}>Review Changes</div>
                  <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
                    The following files will be created in: {fullPreview.troubleshoot_dir}
                  </div>
                </div>
              </div>

              {/* Summary of changes */}
              {fullPreview.parsed_changes && fullPreview.parsed_changes.length > 0 && (
                <div style={{
                  marginBottom: '16px',
                  padding: '12px',
                  background: 'var(--success-bg)',
                  borderRadius: '8px',
                  border: '1px solid var(--success)'
                }}>
                  <div style={{ fontSize: '11px', color: 'var(--success)', marginBottom: '8px', fontWeight: '600', textTransform: 'uppercase' }}>
                    Configuration Changes
                  </div>
                  {fullPreview.parsed_changes.map((change, i) => (
                    <div key={i} style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      marginBottom: '4px',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '12px'
                    }}>
                      <span style={{ color: 'var(--success)' }}>{change.param}</span>
                      <span style={{ color: 'var(--text-tertiary)' }}>=</span>
                      <span style={{ fontWeight: '600' }}>{change.value}</span>
                      <span style={{ color: 'var(--text-tertiary)', fontSize: '10px' }}>
                        ({Math.round(change.confidence * 100)}% confidence)
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* File diffs */}
              {fullPreview.files && Object.entries(fullPreview.files).map(([filename, file]) => (
                <CodeDiff
                  key={filename}
                  filename={filename}
                  original={file.original}
                  proposed={file.proposed}
                  type={file.type}
                />
              ))}

              {/* Skip flags summary */}
              {fullPreview.skip_flags && fullPreview.skip_flags.length > 0 && (
                <div style={{
                  marginTop: '16px',
                  padding: '12px',
                  background: 'var(--bg-tertiary)',
                  borderRadius: '8px',
                  fontSize: '12px'
                }}>
                  <span style={{ color: 'var(--text-tertiary)' }}>Skip flags: </span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--warning)' }}>
                    {fullPreview.skip_flags.join(' ')}
                  </span>
                </div>
              )}
            </div>
          ) : (
            /* Edit Mode - Show form */
            <>
              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '13px', color: 'var(--text-tertiary)', marginBottom: '4px' }}>
                  Original Job
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '14px' }}>
                  {job.job_id} • {job.name}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                  {job.work_dir}
                </div>
              </div>

              {/* AI Recommendations with Apply Fix button */}
              {(recommendations.length > 0 || diagnosisEdits.length > 0) && (
                <div style={{ marginBottom: '16px' }}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '8px'
                  }}>
                    <div style={{ fontSize: '13px', color: 'var(--text-tertiary)' }}>
                      AI Recommendations
                    </div>
                    {(diagnosisEdits.length > 0 || loadingEdits) && (
                      <button
                        onClick={applyAIFix}
                        disabled={loadingEdits || diagnosisEdits.length === 0}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                          padding: '4px 10px',
                          background: loadingEdits ? 'var(--bg-tertiary)' : 'var(--primary)',
                          color: loadingEdits ? 'var(--text-tertiary)' : 'white',
                          border: 'none',
                          borderRadius: '4px',
                          fontSize: '12px',
                          cursor: loadingEdits ? 'wait' : 'pointer',
                          fontWeight: '500'
                        }}
                      >
                        {loadingEdits ? (
                          <Loader2 size={12} className="animate-spin" />
                        ) : (
                          <Zap size={12} />
                        )}
                        {loadingEdits ? 'Loading...' : `Apply AI Fix (${diagnosisEdits.length})`}
                      </button>
                    )}
                  </div>
                  <div style={{
                    background: 'var(--bg-tertiary)',
                    borderRadius: '8px',
                    padding: '12px',
                    fontSize: '13px'
                  }}>
                    {/* Show extracted structured edits */}
                    {diagnosisEdits.length > 0 && (
                      <div style={{ marginBottom: recommendations.length > 0 ? '12px' : 0 }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginBottom: '6px', textTransform: 'uppercase' }}>
                          Detected Changes
                        </div>
                        {diagnosisEdits.map((edit, i) => (
                          <div key={i} style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            marginBottom: i < diagnosisEdits.length - 1 ? '6px' : 0,
                            padding: '4px 8px',
                            background: 'var(--bg-secondary)',
                            borderRadius: '4px'
                          }}>
                            <span style={{
                              padding: '2px 6px',
                              background: edit.confidence > 0.8 ? 'var(--success)' : 'var(--warning)',
                              color: 'white',
                              borderRadius: '3px',
                              fontSize: '10px',
                              fontWeight: '600'
                            }}>
                              {Math.round(edit.confidence * 100)}%
                            </span>
                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
                              {edit.param}
                            </span>
                            <span style={{ color: 'var(--text-tertiary)' }}>→</span>
                            <span style={{ color: 'var(--primary)', fontWeight: '500' }}>
                              {edit.value || edit.action}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                    {/* Show text recommendations */}
                    {recommendations.map((rec, i) => (
                      <div key={i} style={{
                        display: 'flex',
                        gap: '8px',
                        marginBottom: i < recommendations.length - 1 ? '8px' : 0
                      }}>
                        <span style={{ color: 'var(--warning)' }}>•</span>
                        <span>{rec}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Continue From selector */}
              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '13px', color: 'var(--text-tertiary)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Play size={14} />
                  Continue From
                  {loadingFlags && (
                    <Loader2 size={12} className="animate-spin" style={{ color: 'var(--primary)' }} />
                  )}
                  {pipelineFlags.length > 0 && (
                    <span style={{ fontSize: '10px', color: 'var(--success)', padding: '2px 6px', background: 'var(--success-bg)', borderRadius: '3px' }}>
                      Dynamic
                    </span>
                  )}
                  {suggestedPoint && suggestedPoint !== continueFrom && (
                    <span style={{
                      marginLeft: '8px',
                      fontSize: '11px',
                      color: 'var(--warning)',
                      cursor: 'pointer'
                    }} onClick={() => setContinueFrom(suggestedPoint)}>
                      (Suggested: {pipelineSteps.find(s => s.id === suggestedPoint)?.label})
                    </span>
                  )}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {pipelineSteps.map(step => (
                    <label
                      key={step.id}
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '8px',
                        padding: '10px 12px',
                        background: continueFrom === step.id ? 'var(--primary-light)' : 'var(--bg-tertiary)',
                        border: continueFrom === step.id ? '1px solid var(--primary)' : '1px solid var(--border-color)',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        transition: 'all 0.15s'
                      }}
                    >
                      <input
                        type="radio"
                        name="continueFrom"
                        value={step.id}
                        checked={continueFrom === step.id}
                        onChange={(e) => setContinueFrom(e.target.value)}
                        style={{ marginTop: '2px' }}
                      />
                      <div>
                        <div style={{ fontWeight: '500', fontSize: '13px' }}>{step.label}</div>
                        <div style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>{step.description}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '13px', color: 'var(--text-tertiary)', marginBottom: '8px' }}>
                  <Edit3 size={14} style={{ display: 'inline', marginRight: '4px' }} />
                  Describe Modifications
                </div>
                <textarea
                  value={modifications}
                  onChange={(e) => setModifications(e.target.value)}
                  placeholder="e.g., Reduce surfactant count from 400 to 350, increase box size by 10%..."
                  style={{
                    width: '100%',
                    minHeight: '100px',
                    padding: '12px',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '8px',
                    color: 'var(--text-primary)',
                    fontSize: '13px',
                    resize: 'vertical',
                    fontFamily: 'inherit'
                  }}
                />
                <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginTop: '4px' }}>
                  This will create a new directory: troubleshoot/{job.name}-{job.job_id}/
                </div>
              </div>

              {/* Preview Panel */}
              {modifications.trim() && (
                <div style={{ marginBottom: '16px' }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: '8px'
                  }}>
                    <div style={{ fontSize: '13px', color: 'var(--text-tertiary)' }}>
                      <Settings size={14} style={{ display: 'inline', marginRight: '4px' }} />
                      Config Preview
                    </div>
                    {previewLoading && (
                      <Loader2 size={14} className="animate-spin" style={{ color: 'var(--primary)' }} />
                    )}
                  </div>
                  <div style={{
                    background: 'var(--bg-tertiary)',
                    borderRadius: '8px',
                    padding: '12px',
                    border: preview?.validation?.valid === false ? '1px solid var(--error)' : '1px solid var(--border-color)'
                  }}>
                    {preview ? (
                      <>
                        {/* Parsed changes */}
                        {preview.parsed_changes && preview.parsed_changes.length > 0 ? (
                          <div style={{ marginBottom: '12px' }}>
                            <div style={{ fontSize: '11px', color: 'var(--success)', marginBottom: '6px', fontWeight: '600' }}>
                              DETECTED MODIFICATIONS
                            </div>
                            {preview.parsed_changes.map((change, i) => (
                              <div key={i} style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '8px',
                                marginBottom: i < preview.parsed_changes.length - 1 ? '4px' : 0,
                                fontFamily: 'var(--font-mono)',
                                fontSize: '12px'
                              }}>
                                <span style={{
                                  padding: '2px 6px',
                                  background: change.confidence > 0.8 ? 'var(--success)' : 'var(--warning)',
                                  color: 'white',
                                  borderRadius: '3px',
                                  fontSize: '10px'
                                }}>
                                  {Math.round(change.confidence * 100)}%
                                </span>
                                <span style={{ color: 'var(--primary)' }}>{change.param}</span>
                                <span style={{ color: 'var(--text-tertiary)' }}>=</span>
                                <span style={{ color: 'var(--success)' }}>{change.value}</span>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div style={{ color: 'var(--text-tertiary)', fontSize: '12px', marginBottom: '8px' }}>
                            No structured modifications detected. Manual config editing may be needed.
                          </div>
                        )}

                        {/* Validation warnings */}
                        {preview.validation?.warnings?.length > 0 && (
                          <div style={{ marginTop: '8px' }}>
                            {preview.validation.warnings.map((warning, i) => (
                              <div key={i} style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                fontSize: '11px',
                                color: 'var(--warning)',
                                marginBottom: '2px'
                              }}>
                                <AlertTriangle size={12} />
                                {warning}
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Validation errors */}
                        {preview.validation?.errors?.length > 0 && (
                          <div style={{ marginTop: '8px' }}>
                            {preview.validation.errors.map((error, i) => (
                              <div key={i} style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                fontSize: '11px',
                                color: 'var(--error)',
                                marginBottom: '2px'
                              }}>
                                <XCircle size={12} />
                                {error}
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    ) : (
                      <div style={{ color: 'var(--text-tertiary)', fontSize: '12px' }}>
                        {previewLoading ? 'Parsing modifications...' : 'Enter modifications to see preview'}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <div style={{
          padding: '16px 20px',
          borderTop: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'flex-end',
          gap: '8px'
        }}>
          {result ? (
            <button
              onClick={onClose}
              style={{
                padding: '10px 20px',
                background: 'var(--primary)',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontWeight: '500'
              }}
            >
              Done
            </button>
          ) : showCodeReview ? (
            /* Code Review Mode - Show Back/Confirm buttons */
            <>
              <button
                onClick={() => setShowCodeReview(false)}
                style={{
                  padding: '10px 20px',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
              >
                <ChevronUp size={14} style={{ transform: 'rotate(-90deg)' }} />
                Back
              </button>
              <button
                onClick={handleSubmit}
                disabled={submitting}
                style={{
                  padding: '10px 20px',
                  background: submitting ? 'var(--bg-tertiary)' : 'var(--success)',
                  color: submitting ? 'var(--text-tertiary)' : 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: submitting ? 'not-allowed' : 'pointer',
                  fontWeight: '500',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
              >
                {submitting ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    Submitting...
                  </>
                ) : (
                  <>
                    <CheckCircle size={14} />
                    Confirm & Submit
                  </>
                )}
              </button>
            </>
          ) : (
            /* Edit Mode - Show Cancel/Review buttons */
            <>
              <button
                onClick={onClose}
                style={{
                  padding: '10px 20px',
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleReviewCode}
                disabled={loadingFullPreview || !modifications.trim()}
                style={{
                  padding: '10px 20px',
                  background: loadingFullPreview || !modifications.trim() ? 'var(--bg-tertiary)' : 'var(--warning)',
                  color: loadingFullPreview || !modifications.trim() ? 'var(--text-tertiary)' : 'black',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: loadingFullPreview || !modifications.trim() ? 'not-allowed' : 'pointer',
                  fontWeight: '500',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
              >
                {loadingFullPreview ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    Loading Preview...
                  </>
                ) : (
                  <>
                    <FileText size={14} />
                    Review Code
                  </>
                )}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// Diagnosis Modal with Chat
function DiagnosisModal({ job, onClose, onClearDiagnosis }) {
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages]);

  const sendChatMessage = async () => {
    if (!chatInput.trim() || chatLoading) return;

    const question = chatInput.trim();
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', content: question }]);
    setChatLoading(true);

    try {
      const response = await fetch(`${API_URL}/jobs/${job.job_id}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, include_logs: true })
      });
      const data = await response.json();
      setChatMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (e) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: `Error: ${e.message}` }]);
    }
    setChatLoading(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  };

  if (!job || !job.diagnosis) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0, 0, 0, 0.6)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 2000,
        animation: 'fadeIn 0.15s ease-out'
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--bg-secondary)',
          borderRadius: '12px',
          maxWidth: '800px',
          maxHeight: '85vh',
          width: '90%',
          overflow: 'hidden',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
          animation: 'slideUp 0.2s ease-out',
          display: 'flex',
          flexDirection: 'column'
        }}
      >
        {/* Header */}
        <div style={{
          padding: '16px 20px',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexShrink: 0
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Zap size={18} style={{
                color: (job.has_individual_diagnosis && job.has_bulk_diagnosis)
                  ? '#8b5cf6'
                  : job.has_bulk_diagnosis ? '#f97316' : '#8b5cf6'
              }} />
              <span style={{ fontWeight: '600', fontSize: '16px' }}>AI Diagnosis</span>
              {job.has_individual_diagnosis && (
                <span style={{
                  background: 'rgba(139, 92, 246, 0.15)',
                  color: '#8b5cf6',
                  padding: '2px 8px',
                  borderRadius: '10px',
                  fontSize: '11px',
                  fontWeight: '500'
                }}>
                  Individual
                </span>
              )}
              {job.has_bulk_diagnosis && (
                <span style={{
                  background: 'rgba(249, 115, 22, 0.15)',
                  color: '#f97316',
                  padding: '2px 8px',
                  borderRadius: '10px',
                  fontSize: '11px',
                  fontWeight: '500'
                }}>
                  Bulk ({job.diagnosis_batch_job_ids?.length || 0} jobs)
                </span>
              )}
              {job.diagnosis_count > 1 && (
                <span style={{
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-secondary)',
                  padding: '2px 8px',
                  borderRadius: '10px',
                  fontSize: '11px'
                }}>
                  Run #{job.diagnosis_count}
                </span>
              )}
            </div>
            <div style={{ fontSize: '13px', color: 'var(--text-tertiary)', marginTop: '4px' }}>
              Job {job.job_id} • {job.name} • {job.state}
              {job.has_bulk_diagnosis && job.diagnosis_batch_job_ids && (
                <span style={{ marginLeft: '8px', color: '#f97316' }}>
                  • Analyzed with: {job.diagnosis_batch_job_ids.filter(id => id !== job.job_id).slice(0, 3).join(', ')}
                  {job.diagnosis_batch_job_ids.length > 4 && ` +${job.diagnosis_batch_job_ids.length - 4} more`}
                </span>
              )}
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => setShowChat(!showChat)}
              style={{
                background: showChat ? 'var(--diagnosed)' : 'var(--bg-tertiary)',
                border: 'none',
                borderRadius: '6px',
                padding: '8px 12px',
                cursor: 'pointer',
                color: showChat ? 'white' : 'var(--text-secondary)',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '13px'
              }}
              title="Ask follow-up questions"
            >
              <MessageCircle size={16} />
              Chat
            </button>
            {onClearDiagnosis && (
              confirmClear ? (
                <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
                  <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>Clear?</span>
                  <button
                    onClick={() => { onClearDiagnosis(job.job_id); onClose(); }}
                    style={{
                      background: 'var(--error)',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      padding: '4px 8px',
                      cursor: 'pointer',
                      fontSize: '11px'
                    }}
                  >
                    Yes
                  </button>
                  <button
                    onClick={() => setConfirmClear(false)}
                    style={{
                      background: 'var(--bg-tertiary)',
                      border: 'none',
                      borderRadius: '4px',
                      padding: '4px 8px',
                      cursor: 'pointer',
                      fontSize: '11px',
                      color: 'var(--text-secondary)'
                    }}
                  >
                    No
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setConfirmClear(true)}
                  style={{
                    background: 'var(--bg-tertiary)',
                    border: 'none',
                    borderRadius: '6px',
                    padding: '8px 12px',
                    cursor: 'pointer',
                    color: 'var(--error)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    fontSize: '13px'
                  }}
                  title="Clear this diagnosis"
                >
                  <Trash2 size={14} />
                  Clear
                </button>
              )
            )}
            <button
              onClick={onClose}
              style={{
                background: 'var(--bg-tertiary)',
                border: 'none',
                borderRadius: '6px',
                padding: '8px',
                cursor: 'pointer',
                color: 'var(--text-secondary)'
              }}
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          minHeight: 0
        }}>
          {/* Diagnosis */}
          <div style={{
            padding: '20px',
            fontSize: '14px',
            lineHeight: '1.6',
            color: 'var(--text-primary)',
            whiteSpace: 'pre-wrap',
            borderBottom: showChat ? '1px solid var(--border-color)' : 'none'
          }}>
            {job.diagnosis}
          </div>

          {/* Chat Messages */}
          {showChat && chatMessages.length > 0 && (
            <div style={{ padding: '16px 20px' }}>
              {chatMessages.map((msg, idx) => (
                <div
                  key={idx}
                  style={{
                    marginBottom: '12px',
                    display: 'flex',
                    justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start'
                  }}
                >
                  <div style={{
                    maxWidth: '80%',
                    padding: '10px 14px',
                    borderRadius: '12px',
                    background: msg.role === 'user' ? 'var(--diagnosed)' : 'var(--bg-tertiary)',
                    color: msg.role === 'user' ? 'white' : 'var(--text-primary)',
                    fontSize: '13px',
                    lineHeight: '1.5',
                    whiteSpace: 'pre-wrap'
                  }}>
                    {msg.content}
                  </div>
                </div>
              ))}
              {chatLoading && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-tertiary)' }}>
                  <Loader2 size={14} className="animate-spin" />
                  <span style={{ fontSize: '13px' }}>Thinking...</span>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        {/* Chat Input */}
        {showChat && (
          <div style={{
            padding: '16px 20px',
            borderTop: '1px solid var(--border-color)',
            background: 'var(--bg-tertiary)',
            flexShrink: 0
          }}>
            <div style={{ display: 'flex', gap: '10px' }}>
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a follow-up question... (e.g., 'What distance does frame 25 correspond to?')"
                style={{
                  flex: 1,
                  padding: '10px 14px',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '13px',
                  outline: 'none'
                }}
                disabled={chatLoading}
              />
              <button
                onClick={sendChatMessage}
                disabled={!chatInput.trim() || chatLoading}
                style={{
                  padding: '10px 16px',
                  borderRadius: '8px',
                  border: 'none',
                  background: chatInput.trim() && !chatLoading ? 'var(--diagnosed)' : 'var(--bg-secondary)',
                  color: chatInput.trim() && !chatLoading ? 'white' : 'var(--text-tertiary)',
                  cursor: chatInput.trim() && !chatLoading ? 'pointer' : 'not-allowed',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
              >
                <Send size={16} />
              </button>
            </div>
            <div style={{
              marginTop: '8px',
              fontSize: '11px',
              color: 'var(--text-tertiary)'
            }}>
              Ask about frames, distances, progress, or request analysis code snippets
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Troubleshoot History Badge Component
function TroubleshootHistoryBadge({ job }) {
  const [lineage, setLineage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showPopup, setShowPopup] = useState(false);

  const fetchLineage = useCallback(async () => {
    if (lineage || loading) return;
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/jobs/${job.job_id}/lineage`);
      const data = await response.json();
      setLineage(data);
    } catch (e) {
      console.error('Failed to fetch lineage:', e);
    }
    setLoading(false);
  }, [job.job_id, lineage, loading]);

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <button
        onMouseEnter={() => { fetchLineage(); setShowPopup(true); }}
        onMouseLeave={() => setShowPopup(false)}
        onClick={(e) => { e.stopPropagation(); fetchLineage(); setShowPopup(!showPopup); }}
        style={{
          background: 'var(--bg-tertiary)',
          border: '1px solid var(--border-color)',
          borderRadius: '4px',
          padding: '2px 6px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          fontSize: '10px',
          color: 'var(--text-secondary)'
        }}
        title="View troubleshoot history"
      >
        <GitBranch size={10} />
        {loading ? '...' : (lineage?.total_attempts || 0)}
      </button>

      {/* Popup */}
      {showPopup && lineage && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            marginTop: '4px',
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            padding: '12px',
            minWidth: '250px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            zIndex: 100
          }}
          onMouseEnter={() => setShowPopup(true)}
          onMouseLeave={() => setShowPopup(false)}
        >
          <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginBottom: '8px', textTransform: 'uppercase' }}>
            Troubleshoot History
          </div>

          {/* Parent job */}
          {lineage.parent && (
            <div style={{ marginBottom: '8px' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>Parent Job:</div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '4px 8px',
                background: 'var(--bg-tertiary)',
                borderRadius: '4px',
                marginTop: '4px'
              }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
                  {lineage.parent.job_id}
                </span>
                <span className={`status-badge status-${getStateColor(lineage.parent.state)}`} style={{ fontSize: '9px', padding: '1px 4px' }}>
                  {lineage.parent.state}
                </span>
              </div>
            </div>
          )}

          {/* Troubleshoot attempts */}
          {lineage.troubleshoot_history && lineage.troubleshoot_history.length > 0 ? (
            <div>
              <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginBottom: '4px' }}>
                Troubleshoot Attempts:
              </div>
              {lineage.troubleshoot_history.map((attempt, i) => (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '8px',
                    padding: '4px 8px',
                    background: 'var(--bg-tertiary)',
                    borderRadius: '4px',
                    marginBottom: '4px',
                    borderLeft: `3px solid ${attempt.state === 'COMPLETED' ? 'var(--success)' : attempt.state === 'FAILED' ? 'var(--error)' : 'var(--warning)'}`
                  }}
                >
                  <div>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
                      #{attempt.attempt}
                    </span>
                    <span style={{ color: 'var(--text-tertiary)', marginLeft: '4px', fontSize: '10px' }}>
                      {attempt.job_id}
                    </span>
                  </div>
                  <span className={`status-badge status-${getStateColor(attempt.state)}`} style={{ fontSize: '9px', padding: '1px 4px' }}>
                    {attempt.state}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            !lineage.parent && (
              <div style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
                No troubleshoot history
              </div>
            )
          )}

          {/* Modifications applied */}
          {lineage.modifications_applied && (
            <div style={{ marginTop: '8px' }}>
              <div style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>Modifications:</div>
              <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                {lineage.modifications_applied.text || 'N/A'}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Job Card Component
function JobCard({ job, onRemove, onDiagnose, onViewDiagnosis, onTroubleshoot, onMoveToNotes, selectMode, isSelected, onToggleSelect }) {
  const [expanded, setExpanded] = useState(false);

  const hasDiagnosis = job.diagnosis_count > 0 || job.diagnosis;

  return (
    <div
      className={`job-card animate-slide-in ${job.needs_attention ? 'needs-attention' : ''} ${hasDiagnosis ? 'diagnosed' : ''} ${isSelected ? 'selected' : ''}`}
      style={isSelected ? { borderColor: 'var(--primary)', boxShadow: '0 0 0 2px var(--primary-glow)' } : {}}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
          {/* Selection checkbox */}
          {selectMode && (
            <button
              onClick={(e) => { e.stopPropagation(); onToggleSelect(job.job_id); }}
              style={{
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                padding: '2px',
                color: isSelected ? 'var(--primary)' : 'var(--text-tertiary)'
              }}
            >
              {isSelected ? <CheckSquare size={20} /> : <Square size={20} />}
            </button>
          )}
          <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="mono" style={{ fontSize: '18px', fontWeight: '600' }}>
              {job.job_id}
            </span>
            <span className={`status-badge status-${getStateColor(job.state)}`}>
              {getStateIcon(job.state)}
              {job.state}
            </span>
            {hasDiagnosis && (
              <button
                onClick={(e) => { e.stopPropagation(); onViewDiagnosis(job); }}
                style={{
                  background: (job.has_individual_diagnosis && job.has_bulk_diagnosis)
                    ? 'linear-gradient(180deg, #8b5cf6 50%, #f97316 50%)'
                    : job.has_bulk_diagnosis ? '#f97316' : '#8b5cf6',
                  color: 'white',
                  padding: '2px 8px',
                  borderRadius: '4px',
                  fontSize: '10px',
                  fontWeight: '500',
                  border: 'none',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  position: 'relative'
                }}
                title={(job.has_individual_diagnosis && job.has_bulk_diagnosis)
                  ? 'Both individual and bulk AI diagnoses performed'
                  : job.has_bulk_diagnosis
                    ? `Bulk AI Diagnosis (${job.diagnosis_batch_job_ids?.length || 0} jobs analyzed together)`
                    : `Individual AI Diagnosis (${job.diagnosis_count || 1} run${(job.diagnosis_count || 1) > 1 ? 's' : ''})`}
              >
                <Zap size={10} />
                AI
                {(job.diagnosis_count || 1) > 1 && (
                  <span style={{
                    position: 'absolute',
                    top: '-6px',
                    right: '-6px',
                    background: 'var(--error)',
                    color: 'white',
                    borderRadius: '50%',
                    width: '14px',
                    height: '14px',
                    fontSize: '9px',
                    fontWeight: '600',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    {job.diagnosis_count}
                  </span>
                )}
              </button>
            )}
            {/* Troubleshoot history badge */}
            <TroubleshootHistoryBadge job={job} />
          </div>
          <div style={{ fontSize: '14px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            {job.name}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginTop: '2px' }}>
            {job.cluster}
          </div>
          </div>
        </div>
        <button
          onClick={() => onRemove(job.job_id)}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-tertiary)',
            cursor: 'pointer',
            padding: '4px'
          }}
          title="Remove from dashboard"
        >
          <X size={16} />
        </button>
      </div>

      {/* Progress */}
      {job.state === 'RUNNING' && (
        <>
          <ProgressBar progress={job.progress} state={job.state} />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px', fontSize: '12px', color: 'var(--text-tertiary)' }}>
            <span>{job.progress.toFixed(1)}% complete</span>
            <span>{job.time_elapsed} / {job.time_limit}</span>
          </div>
        </>
      )}

      {/* Quick Stats */}
      <div style={{
        display: 'flex',
        gap: '16px',
        marginTop: '12px',
        fontSize: '13px',
        color: 'var(--text-secondary)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Layers size={14} />
          <span>{job.nodes} node{job.nodes > 1 ? 's' : ''}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Cpu size={14} />
          <span>{job.cpus} CPU{job.cpus > 1 ? 's' : ''}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <Clock size={14} />
          <span className="mono">{formatTime(job.time_elapsed)}</span>
        </div>
      </div>

      {/* Expandable Details */}
      <button
        onClick={() => setExpanded(!expanded)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          background: 'none',
          border: 'none',
          color: 'var(--primary)',
          cursor: 'pointer',
          padding: '8px 0',
          fontSize: '12px',
          marginTop: '8px'
        }}
      >
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        {expanded ? 'Hide details' : 'Show details'}
      </button>

      {expanded && (
        <div style={{
          marginTop: '8px',
          padding: '12px',
          background: 'var(--bg-tertiary)',
          borderRadius: '8px',
          fontSize: '13px'
        }}>
          <div style={{ display: 'grid', gap: '8px' }}>
            <div>
              <span style={{ color: 'var(--text-tertiary)' }}>Partition: </span>
              <span className="mono">{job.partition || 'N/A'}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-tertiary)' }}>Work Dir: </span>
              <span className="mono" style={{ fontSize: '11px', wordBreak: 'break-all' }}>
                {job.work_dir || 'N/A'}
              </span>
            </div>
            <div>
              <span style={{ color: 'var(--text-tertiary)' }}>Time Limit: </span>
              <span className="mono">{job.time_limit}</span>
            </div>
            {job.submit_time && (
              <div>
                <span style={{ color: 'var(--text-tertiary)' }}>Submitted: </span>
                <span className="mono">{new Date(job.submit_time).toLocaleString()}</span>
              </div>
            )}
            {job.start_time && (
              <div>
                <span style={{ color: 'var(--text-tertiary)' }}>Started: </span>
                <span className="mono">{new Date(job.start_time).toLocaleString()}</span>
              </div>
            )}
            {job.last_update && (
              <div>
                <span style={{ color: 'var(--text-tertiary)' }}>Last Update: </span>
                <span className="mono">{new Date(job.last_update).toLocaleString()}</span>
              </div>
            )}
          </div>

          {/* Warnings/Errors */}
          {job.warnings && job.warnings.length > 0 && (
            <div style={{ marginTop: '12px' }}>
              <div style={{ color: 'var(--warning)', fontWeight: '500', marginBottom: '4px' }}>
                <AlertTriangle size={14} style={{ display: 'inline', marginRight: '4px' }} />
                Warnings
              </div>
              {job.warnings.map((w, i) => (
                <div key={i} style={{ fontSize: '12px', color: 'var(--text-secondary)', marginLeft: '18px' }}>
                  {w}
                </div>
              ))}
            </div>
          )}

          {job.errors && job.errors.length > 0 && (
            <div style={{ marginTop: '12px' }}>
              <div style={{ color: 'var(--error)', fontWeight: '500', marginBottom: '4px' }}>
                <XCircle size={14} style={{ display: 'inline', marginRight: '4px' }} />
                Errors
              </div>
              {job.errors.map((e, i) => (
                <div key={i} style={{ fontSize: '12px', color: 'var(--text-secondary)', marginLeft: '18px' }}>
                  {e}
                </div>
              ))}
            </div>
          )}

          {/* Diagnosis */}
          {job.diagnosis && (
            <div style={{
              marginTop: '12px',
              padding: '12px',
              background: 'var(--bg-secondary)',
              borderRadius: '6px',
              borderLeft: '3px solid var(--primary)',
              maxHeight: '300px',
              overflowY: 'auto'
            }}>
              <div style={{ color: 'var(--primary)', fontWeight: '500', marginBottom: '8px', fontSize: '12px' }}>
                <Zap size={14} style={{ display: 'inline', marginRight: '4px' }} />
                AI Diagnosis
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>
                {job.diagnosis}
              </div>
            </div>
          )}

          {/* Actions - Show diagnosis/check progress for all jobs */}
          <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
            {!job.diagnosis && (
              <button
                onClick={() => onDiagnose(job.job_id)}
                disabled={job.diagnosing}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '8px 12px',
                  background: job.diagnosing ? 'var(--bg-tertiary)' :
                             job.state === 'RUNNING' ? 'var(--running)' : 'var(--primary)',
                  color: job.diagnosing ? 'var(--text-secondary)' : 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: job.diagnosing ? 'not-allowed' : 'pointer',
                  fontSize: '13px',
                  fontWeight: '500'
                }}
              >
                {job.diagnosing ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    {job.state === 'RUNNING' ? 'Checking...' : 'Diagnosing...'}
                  </>
                ) : (
                  <>
                    {job.state === 'RUNNING' ? <Activity size={14} /> : <Terminal size={14} />}
                    {job.state === 'RUNNING' ? 'Check Progress' :
                     job.state === 'PENDING' ? 'Check Status' : 'Run Diagnosis'}
                  </>
                )}
              </button>
            )}
            {job.diagnosis && (
              <>
                <button
                  onClick={() => onViewDiagnosis(job)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '8px 12px',
                    background: 'var(--primary)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '13px',
                    fontWeight: '500'
                  }}
                >
                  <Zap size={14} />
                  View Full
                </button>
                <button
                  onClick={() => onDiagnose(job.job_id, true)}
                  disabled={job.diagnosing}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '8px 12px',
                    background: 'var(--bg-tertiary)',
                    color: 'var(--text-secondary)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    cursor: job.diagnosing ? 'not-allowed' : 'pointer',
                    fontSize: '13px'
                  }}
                >
                  <RefreshCw size={14} className={job.diagnosing ? 'animate-spin' : ''} />
                  {job.state === 'RUNNING' ? 'Refresh' : 'Re-analyze'}
                </button>
                {/* Troubleshoot button for failed/completed jobs with diagnosis */}
                {!['RUNNING', 'PENDING'].includes(job.state) && (
                  <button
                    onClick={() => onTroubleshoot(job)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '8px 12px',
                      background: 'var(--warning)',
                      color: 'black',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '13px',
                      fontWeight: '500'
                    }}
                  >
                    <GitBranch size={14} />
                    Troubleshoot
                  </button>
                )}
                {/* Move to Notes button */}
                {!['RUNNING', 'PENDING'].includes(job.state) && onMoveToNotes && (
                  <button
                    onClick={() => onMoveToNotes(job)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '8px 12px',
                      background: '#f97316',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '13px',
                      fontWeight: '500'
                    }}
                    title="Archive job and create a note"
                  >
                    <StickyNote size={14} />
                    To Notes
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Empty State
function EmptyState({ hasFilters, onClearFilters }) {
  return (
    <div className="empty-state">
      <Server size={64} className="empty-state-icon" />
      <h3 style={{ marginBottom: '8px', color: 'var(--text-secondary)' }}>
        {hasFilters ? 'No Matching Jobs' : 'No Jobs Found'}
      </h3>
      <p>
        {hasFilters
          ? 'No jobs match your current filters.'
          : 'Jobs will appear here as they are detected on the cluster.'}
      </p>
      {hasFilters && (
        <button
          onClick={onClearFilters}
          style={{
            marginTop: '16px',
            padding: '8px 16px',
            background: 'var(--primary)',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer'
          }}
        >
          Clear Filters
        </button>
      )}
    </div>
  );
}

// ============================================================
// Main App Component
// ============================================================

function App() {
  const [jobs, setJobs] = useState([]);
  const [clusters, setClusters] = useState({});
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [lastUpdate, setLastUpdate] = useState(null);
  const [statusFilter, setStatusFilter] = useState(null);
  const [clusterFilter, setClusterFilter] = useState(null);
  const [workDirFilter, setWorkDirFilter] = useState(null);
  const [sortBy, setSortBy] = useState('time_desc'); // time_desc, time_asc, id_desc, id_asc, name_asc, name_desc
  const [showDiagnosisLog, setShowDiagnosisLog] = useState(false);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  const [diagnosisLog, setDiagnosisLog] = useState([]);
  const [diagnosisModal, setDiagnosisModal] = useState(null);  // Job to show in modal
  const [troubleshootModal, setTroubleshootModal] = useState(null);  // Job to troubleshoot
  // Multi-select and batch operations
  const [selectedJobs, setSelectedJobs] = useState(new Set());
  const [selectMode, setSelectMode] = useState(false);
  const [batchOperationStatus, setBatchOperationStatus] = useState(null);
  const [crossJobSummary, setCrossJobSummary] = useState(null);
  const [errorClusters, setErrorClusters] = useState([]);
  const [bulkChatMessages, setBulkChatMessages] = useState([]);
  const [bulkChatInput, setBulkChatInput] = useState('');
  const [bulkChatLoading, setBulkChatLoading] = useState(false);
  const [showPasteModal, setShowPasteModal] = useState(false);
  const [pasteInput, setPasteInput] = useState('');
  const [parsedJobIds, setParsedJobIds] = useState([]);
  // Project notes
  const [showNotes, setShowNotes] = useState(false);
  const [projectNotes, setProjectNotes] = useState([]);
  const [newNote, setNewNote] = useState({ note: '', category: 'general', job_names: '' });
  const [showAddNote, setShowAddNote] = useState(false);
  // Move to Notes modal
  const [moveToNotesJob, setMoveToNotesJob] = useState(null);
  const [moveToNotesText, setMoveToNotesText] = useState('');
  const [moveToNotesCategory, setMoveToNotesCategory] = useState('failure');
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  // WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setConnectionStatus('connecting');
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case 'initial':
            setJobs(data.jobs || []);
            setClusters(data.clusters || {});
            setLastUpdate(new Date());
            // Initialize diagnosis log from jobs with diagnoses
            const initialLog = (data.jobs || [])
              .filter(j => j.diagnosis && !j.diagnosis.startsWith('Debug agent not available'))
              .map(j => ({
                job_id: j.job_id,
                diagnosis: j.diagnosis,
                timestamp: new Date().toLocaleString()
              }));
            if (initialLog.length > 0) {
              setDiagnosisLog(initialLog);
            }
            break;

          case 'jobs_update':
            setJobs(prevJobs => {
              const jobMap = new Map(prevJobs.map(j => [j.job_id, j]));

              for (const job of data.jobs) {
                const existing = jobMap.get(job.job_id);
                jobMap.set(job.job_id, { ...existing, ...job });
              }

              return Array.from(jobMap.values());
            });
            setLastUpdate(new Date());
            break;

          case 'diagnosis_complete':
            setJobs(prevJobs => {
              return prevJobs.map(j => {
                if (j.job_id === data.job_id) {
                  return { ...j, diagnosis: data.diagnosis, diagnosing: false };
                }
                return j;
              });
            });
            // Add to diagnosis log
            if (data.diagnosis && !data.diagnosis.startsWith('Debug agent not available')) {
              setDiagnosisLog(prev => [{
                job_id: data.job_id,
                diagnosis: data.diagnosis,
                timestamp: new Date().toLocaleString()
              }, ...prev]);
            }
            setLastUpdate(new Date());
            break;

          case 'error':
            console.error('Server error:', data.message);
            break;

          case 'pong':
            break;

          case 'bulk_diagnose_complete':
            console.log('Bulk diagnose complete:', data);
            setBatchOperationStatus({
              type: 'diagnose',
              status: 'complete',
              results: data.results
            });
            // Update jobs with new diagnoses
            if (data.results) {
              data.results.forEach(result => {
                if (result.status === 'completed') {
                  // Refresh to get updated diagnoses
                  handleRefresh();
                }
              });
            }
            // Update summary if available
            if (data.summary) {
              setCrossJobSummary(data.summary);
            }
            // Auto-clear status after 5 seconds
            setTimeout(() => setBatchOperationStatus(null), 5000);
            break;

          case 'batch_troubleshoot_complete':
            console.log('Batch troubleshoot complete:', data);
            setBatchOperationStatus({
              type: 'troubleshoot',
              status: 'complete',
              results: data.results
            });
            setTimeout(() => setBatchOperationStatus(null), 5000);
            break;

          default:
            console.log('Unknown message type:', data.type);
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnectionStatus('disconnected');
      reconnectTimeoutRef.current = setTimeout(() => {
        connectWebSocket();
      }, 5000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    wsRef.current = ws;
  }, []);

  // Connect on mount
  useEffect(() => {
    connectWebSocket();

    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);

    return () => {
      clearInterval(pingInterval);
      clearTimeout(reconnectTimeoutRef.current);
      wsRef.current?.close();
    };
  }, [connectWebSocket]);

  // Remove job from view
  const handleRemoveJob = async (jobId) => {
    try {
      await fetch(`${API_URL}/jobs/${jobId}`, { method: 'DELETE' });
      setJobs(jobs.filter(j => j.job_id !== jobId));
    } catch (e) {
      console.error('Failed to remove job:', e);
    }
  };

  // Trigger diagnosis (force=true for re-diagnosis)
  const handleDiagnose = async (jobId, force = false) => {
    setJobs(prevJobs => prevJobs.map(j =>
      j.job_id === jobId ? { ...j, diagnosing: true } : j
    ));

    try {
      const response = await fetch(`${API_URL}/jobs/${jobId}/diagnose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId, force: force })
      });
      const data = await response.json();
      console.log('Diagnosis response:', data);

      if (data.status === 'complete' && data.diagnosis) {
        setJobs(prevJobs => prevJobs.map(j =>
          j.job_id === jobId ? { ...j, diagnosis: data.diagnosis, diagnosing: false } : j
        ));
        // Add to log
        if (!data.diagnosis.startsWith('Debug agent not available')) {
          setDiagnosisLog(prev => [{
            job_id: jobId,
            diagnosis: data.diagnosis,
            timestamp: new Date().toLocaleString()
          }, ...prev]);
        }
      }
    } catch (e) {
      console.error('Failed to queue diagnosis:', e);
      setJobs(prevJobs => prevJobs.map(j =>
        j.job_id === jobId ? { ...j, diagnosing: false } : j
      ));
    }
  };

  // View diagnosis in modal
  const handleViewDiagnosis = (job) => {
    setDiagnosisModal(job);
  };

  // Open troubleshoot modal
  const handleTroubleshoot = (job) => {
    setTroubleshootModal(job);
  };

  // Manual refresh
  const handleRefresh = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'refresh' }));
    }
  };

  const handleClearDiagnosis = async (jobId) => {
    try {
      await fetch(`${API_URL}/jobs/${jobId}/diagnosis`, { method: 'DELETE' });
      setDiagnosisModal(null);
      handleRefresh();
    } catch (err) {
      console.error('Failed to clear diagnosis:', err);
    }
  };

  // Fetch project notes
  const fetchNotes = async () => {
    try {
      const res = await fetch(`${API_URL}/notes`);
      const data = await res.json();
      setProjectNotes(data.notes || []);
    } catch (e) {
      console.error('Failed to fetch notes:', e);
    }
  };

  // Add a new note
  const handleAddNote = async () => {
    if (!newNote.note.trim()) return;
    try {
      const jobNames = newNote.job_names.split(',').map(s => s.trim()).filter(s => s);
      await fetch(`${API_URL}/notes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project: 'default',
          note: newNote.note,
          category: newNote.category,
          job_names: jobNames
        })
      });
      setNewNote({ note: '', category: 'general', job_names: '' });
      setShowAddNote(false);
      fetchNotes();
    } catch (e) {
      console.error('Failed to add note:', e);
    }
  };

  // Resolve a note
  const handleResolveNote = async (noteId) => {
    try {
      await fetch(`${API_URL}/notes/${noteId}/resolve`, { method: 'POST' });
      fetchNotes();
    } catch (e) {
      console.error('Failed to resolve note:', e);
    }
  };

  // Delete a note
  const handleDeleteNote = async (noteId) => {
    try {
      await fetch(`${API_URL}/notes/${noteId}`, { method: 'DELETE' });
      fetchNotes();
    } catch (e) {
      console.error('Failed to delete note:', e);
    }
  };

  // Open Move to Notes modal
  const openMoveToNotes = (job) => {
    setMoveToNotesJob(job);
    // Pre-fill with diagnosis summary if available
    if (job.diagnosis) {
      // Extract key sections from diagnosis
      const lines = job.diagnosis.split('\n');
      const keyParts = [];
      let inSection = false;
      let currentSection = '';

      for (const line of lines) {
        if (line.includes('**Status**') || line.includes('**Issues Found**') || line.includes('**Recommendations**')) {
          inSection = true;
          currentSection = line;
        } else if (inSection && line.startsWith('###') || line.startsWith('---')) {
          if (currentSection) keyParts.push(currentSection);
          inSection = false;
          currentSection = '';
        } else if (inSection) {
          currentSection += '\n' + line;
        }
      }
      if (currentSection) keyParts.push(currentSection);

      // Default to first 500 chars of diagnosis if no structured sections found
      const summary = keyParts.length > 0 ? keyParts.join('\n\n') : job.diagnosis.substring(0, 500);
      setMoveToNotesText(summary);
    } else {
      setMoveToNotesText(`Job ${job.job_id} (${job.name}) - ${job.state}`);
    }
    setMoveToNotesCategory('failure');
  };

  // Execute move to notes
  const handleMoveToNotes = async () => {
    if (!moveToNotesJob) return;

    try {
      // 1. Create the note
      await fetch(`${API_URL}/notes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project: 'default',
          note: moveToNotesText,
          category: moveToNotesCategory,
          job_names: [moveToNotesJob.name]
        })
      });

      // 2. Hide the job
      await fetch(`${API_URL}/jobs/${moveToNotesJob.job_id}/hide`, { method: 'POST' });

      // 3. Refresh
      setMoveToNotesJob(null);
      setMoveToNotesText('');
      fetchNotes();
      handleRefresh();
    } catch (e) {
      console.error('Failed to move job to notes:', e);
    }
  };

  // Toggle job selection
  const toggleJobSelection = (jobId) => {
    setSelectedJobs(prev => {
      const newSet = new Set(prev);
      if (newSet.has(jobId)) {
        newSet.delete(jobId);
      } else {
        newSet.add(jobId);
      }
      return newSet;
    });
  };

  // Select all failed jobs
  const selectAllFailed = () => {
    const failedIds = jobs.filter(j => j.state === 'FAILED').map(j => j.job_id);
    setSelectedJobs(new Set(failedIds));
    setSelectMode(true);
  };

  // Clear selection
  const clearSelection = () => {
    setSelectedJobs(new Set());
    setSelectMode(false);
  };

  // Bulk diagnose selected jobs
  const handleBulkDiagnose = async () => {
    if (selectedJobs.size === 0) return;

    setBatchOperationStatus({ type: 'diagnose', status: 'running', count: selectedJobs.size });

    try {
      const response = await fetch(`${API_URL}/bulk-diagnose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_ids: Array.from(selectedJobs) })
      });
      const data = await response.json();
      console.log('Bulk diagnose started:', data);
      setBatchOperationStatus({ type: 'diagnose', status: 'started', operation_id: data.operation_id });

      // Poll for completion (in case WebSocket message is missed)
      const pollInterval = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_URL}/batch-operations`);
          const statusData = await statusRes.json();
          const op = statusData.operations?.find(o => o.id === data.operation_id ||
            (o.type === 'bulk_diagnose' && o.status === 'completed' &&
             JSON.stringify(o.job_ids) === JSON.stringify(Array.from(selectedJobs))));

          if (op && op.status === 'completed') {
            clearInterval(pollInterval);
            console.log('Bulk diagnose completed (poll):', op);
            setBatchOperationStatus({ type: 'diagnose', status: 'complete', results: op.result_summary });
            handleRefresh(); // Refresh jobs to show new diagnoses
            setTimeout(() => setBatchOperationStatus(null), 5000);
          }
        } catch (pollErr) {
          console.error('Poll error:', pollErr);
        }
      }, 5000); // Poll every 5 seconds

      // Stop polling after 10 minutes max
      setTimeout(() => clearInterval(pollInterval), 600000);
    } catch (e) {
      console.error('Bulk diagnose failed:', e);
      setBatchOperationStatus({ type: 'diagnose', status: 'error', message: e.message });
    }
  };

  // Batch troubleshoot selected jobs
  const handleBatchTroubleshoot = async (modifications) => {
    if (selectedJobs.size === 0) return;

    setBatchOperationStatus({ type: 'troubleshoot', status: 'running', count: selectedJobs.size });

    try {
      const response = await fetch(`${API_URL}/batch-troubleshoot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_ids: Array.from(selectedJobs),
          modifications: modifications
        })
      });
      const data = await response.json();
      console.log('Batch troubleshoot started:', data);
      setBatchOperationStatus({ type: 'troubleshoot', status: 'started', operation_id: data.operation_id });
      clearSelection();
    } catch (e) {
      console.error('Batch troubleshoot failed:', e);
      setBatchOperationStatus({ type: 'troubleshoot', status: 'error', message: e.message });
    }
  };

  // Fetch cross-job summary
  const fetchSummary = async () => {
    try {
      const response = await fetch(`${API_URL}/summary`);
      const data = await response.json();
      setCrossJobSummary(data);
      setShowSummary(true);
    } catch (e) {
      console.error('Failed to fetch summary:', e);
    }
  };

  // Fetch error clusters
  const fetchErrorClusters = async () => {
    try {
      const response = await fetch(`${API_URL}/patterns`);
      const data = await response.json();
      setErrorClusters(data.patterns || []);
    } catch (e) {
      console.error('Failed to fetch error clusters:', e);
    }
  };

  // Bulk chat - ask questions about multiple jobs
  const handleBulkChat = async () => {
    if (!bulkChatInput.trim()) return;

    const question = bulkChatInput;
    setBulkChatInput('');
    setBulkChatLoading(true);

    // Add user message immediately
    setBulkChatMessages(prev => [...prev, { role: 'user', content: question }]);

    // Get job IDs to analyze - use selected jobs or all failed jobs
    const jobIds = selectedJobs.size > 0
      ? Array.from(selectedJobs)
      : jobs.filter(j => j.state === 'FAILED').map(j => j.job_id);

    try {
      const response = await fetch(`${API_URL}/bulk-chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_ids: jobIds, question })
      });
      const data = await response.json();

      if (data.status === 'success') {
        setBulkChatMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
      } else {
        setBulkChatMessages(prev => [...prev, { role: 'assistant', content: `Error: ${data.detail || 'Unknown error'}` }]);
      }
    } catch (e) {
      setBulkChatMessages(prev => [...prev, { role: 'assistant', content: `Failed to get response: ${e.message}` }]);
    } finally {
      setBulkChatLoading(false);
    }
  };

  // Parse job IDs from pasted text (supports squeue output, "Job X: Name" format, etc.)
  const parseJobIdsFromText = (text) => {
    const jobIds = new Set();
    const lines = text.split('\n');

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      // Pattern 1: squeue format - job ID is first column (e.g., "46042817 gpu-share surf-369...")
      const squeueMatch = trimmed.match(/^\s*(\d{6,10})\s+/);
      if (squeueMatch) {
        jobIds.add(squeueMatch[1]);
        continue;
      }

      // Pattern 2: "Job 46042817: SURF369" format
      const jobColonMatch = trimmed.match(/Job\s+(\d{6,10})/i);
      if (jobColonMatch) {
        jobIds.add(jobColonMatch[1]);
        continue;
      }

      // Pattern 3: Just job IDs, one per line or comma/space separated
      const idsInLine = trimmed.match(/\b(\d{6,10})\b/g);
      if (idsInLine) {
        idsInLine.forEach(id => jobIds.add(id));
      }
    }

    return Array.from(jobIds);
  };

  // Handle paste input change
  const handlePasteInputChange = (text) => {
    setPasteInput(text);
    const ids = parseJobIdsFromText(text);
    setParsedJobIds(ids);
  };

  // Apply parsed job IDs to selection
  const applyParsedJobIds = () => {
    if (parsedJobIds.length === 0) return;

    // Filter to only include job IDs that exist in our jobs list
    const validIds = parsedJobIds.filter(id => jobs.some(j => j.job_id === id));
    setSelectedJobs(new Set(validIds));
    setSelectMode(true);
    setShowPasteModal(false);
    setPasteInput('');
    setParsedJobIds([]);
  };

  // Parse time string to seconds for sorting
  const parseTimeToSeconds = (timeStr) => {
    if (!timeStr) return 0;
    try {
      if (timeStr.includes('-')) {
        const [days, rest] = timeStr.split('-');
        const parts = rest.split(':');
        return parseInt(days) * 86400 + parseInt(parts[0]) * 3600 + parseInt(parts[1] || 0) * 60 + parseInt(parts[2] || 0);
      }
      const parts = timeStr.split(':');
      return parseInt(parts[0]) * 3600 + parseInt(parts[1] || 0) * 60 + parseInt(parts[2] || 0);
    } catch {
      return 0;
    }
  };

  // Filter and sort jobs
  const filteredJobs = jobs
    .filter(job => {
      if (statusFilter && getStateCategory(job.state) !== statusFilter) return false;
      if (clusterFilter && job.cluster !== clusterFilter) return false;
      if (workDirFilter && extractProjectFolder(job.work_dir) !== workDirFilter) return false;
      return true;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'time_desc':
          return parseTimeToSeconds(b.time_elapsed) - parseTimeToSeconds(a.time_elapsed);
        case 'time_asc':
          return parseTimeToSeconds(a.time_elapsed) - parseTimeToSeconds(b.time_elapsed);
        case 'id_desc':
          return parseInt(b.job_id) - parseInt(a.job_id);
        case 'id_asc':
          return parseInt(a.job_id) - parseInt(b.job_id);
        case 'name_asc':
          return a.name.localeCompare(b.name);
        case 'name_desc':
          return b.name.localeCompare(a.name);
        default:
          return 0;
      }
    });

  const hasFilters = statusFilter || clusterFilter || workDirFilter;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <header className="dashboard-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Server size={24} style={{ color: 'var(--primary)' }} />
            <h1 style={{ fontSize: '20px', fontWeight: '600' }}>HPC Monitor</h1>
          </div>
          <ClusterFilter
            clusters={clusters}
            clusterFilter={clusterFilter}
            onClusterFilterChange={setClusterFilter}
          />
          <WorkDirFilter
            jobs={jobs}
            workDirFilter={workDirFilter}
            onWorkDirFilterChange={setWorkDirFilter}
          />
          {/* Sort Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ArrowUpDown size={14} style={{ color: 'var(--text-tertiary)' }} />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              style={{
                padding: '6px 10px',
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--border-color)',
                borderRadius: '6px',
                color: 'var(--text-primary)',
                fontSize: '13px',
                cursor: 'pointer',
                fontFamily: 'var(--font-sans)'
              }}
            >
              <option value="time_desc">Longest Running</option>
              <option value="time_asc">Shortest Running</option>
              <option value="id_desc">Newest First</option>
              <option value="id_asc">Oldest First</option>
              <option value="name_asc">Name A-Z</option>
              <option value="name_desc">Name Z-A</option>
            </select>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <ConnectionStatus status={connectionStatus} />
          {lastUpdate && (
            <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
              {lastUpdate.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={() => setShowDiagnosisLog(!showDiagnosisLog)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 12px',
              background: showDiagnosisLog ? 'var(--primary)' : 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              color: showDiagnosisLog ? 'white' : 'var(--text-primary)',
              cursor: 'pointer',
              fontSize: '13px'
            }}
            title="View diagnosis history"
          >
            <FileText size={14} />
            Log {diagnosisLog.length > 0 && `(${diagnosisLog.length})`}
          </button>
          <button
            onClick={() => setShowAnalytics(!showAnalytics)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 12px',
              background: showAnalytics ? 'var(--primary)' : 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              color: showAnalytics ? 'white' : 'var(--text-primary)',
              cursor: 'pointer',
              fontSize: '13px'
            }}
            title="Usage Analytics"
          >
            <Activity size={14} />
            Analytics
          </button>
          <button
            onClick={handleRefresh}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 12px',
              background: 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              color: 'var(--text-primary)',
              cursor: 'pointer',
              fontSize: '13px'
            }}
          >
            <RefreshCw size={14} />
            Refresh
          </button>
        </div>
      </header>

      {/* Stats Bar with filters */}
      <StatsBar
        jobs={jobs}
        statusFilter={statusFilter}
        onStatusFilterChange={setStatusFilter}
      />

      {/* Batch Actions Toolbar */}
      <div style={{
        padding: '8px 24px',
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border-color)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '12px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button
            onClick={() => setSelectMode(!selectMode)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              background: selectMode ? 'var(--primary)' : 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              color: selectMode ? 'white' : 'var(--text-primary)',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            <CheckSquare size={14} />
            {selectMode ? 'Exit Select' : 'Select Jobs'}
          </button>

          {selectMode && (
            <>
              <button
                onClick={selectAllFailed}
                style={{
                  padding: '6px 12px',
                  background: 'var(--error-bg)',
                  border: '1px solid var(--error)',
                  borderRadius: '6px',
                  color: 'var(--error)',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                Select All Failed
              </button>

              <button
                onClick={() => setShowPasteModal(true)}
                style={{
                  padding: '6px 12px',
                  background: 'var(--bg-tertiary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                Paste IDs
              </button>

              {selectedJobs.size > 0 && (
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                  {selectedJobs.size} selected
                </span>
              )}
            </>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {selectedJobs.size > 0 && (
            <>
              <button
                onClick={handleBulkDiagnose}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '6px 12px',
                  background: 'var(--diagnosed)',
                  border: 'none',
                  borderRadius: '6px',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                <Sparkles size={14} />
                Bulk Diagnose ({selectedJobs.size})
              </button>
              <button
                onClick={clearSelection}
                style={{
                  padding: '6px 12px',
                  background: 'var(--bg-tertiary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                Clear
              </button>
            </>
          )}

          <button
            onClick={fetchSummary}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              background: showSummary ? 'var(--primary)' : 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              color: showSummary ? 'white' : 'var(--text-primary)',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            <BarChart3 size={14} />
            Summary
          </button>

          <button
            onClick={() => { setShowNotes(!showNotes); if (!showNotes) fetchNotes(); }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              background: showNotes ? '#f97316' : 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              color: showNotes ? 'white' : 'var(--text-primary)',
              cursor: 'pointer',
              fontSize: '12px',
              position: 'relative'
            }}
          >
            <StickyNote size={14} />
            Notes
            {projectNotes.length > 0 && (
              <span style={{
                background: showNotes ? 'white' : '#f97316',
                color: showNotes ? '#f97316' : 'white',
                borderRadius: '10px',
                padding: '1px 6px',
                fontSize: '10px',
                fontWeight: '600'
              }}>
                {projectNotes.length}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Cross-Job Summary Panel */}
      {showSummary && crossJobSummary && (
        <div style={{
          padding: '16px 24px',
          background: 'var(--bg-elevated)',
          borderBottom: '1px solid var(--border-color)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3 style={{ fontSize: '16px', fontWeight: '600' }}>Cross-Job Analysis</h3>
            <button
              onClick={() => setShowSummary(false)}
              style={{ background: 'none', border: 'none', color: 'var(--text-tertiary)', cursor: 'pointer' }}
            >
              <X size={16} />
            </button>
          </div>

          <div style={{ display: 'flex', gap: '24px', marginBottom: '16px' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: '600' }}>{crossJobSummary.total_jobs}</div>
              <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>Total Jobs</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: '600', color: 'var(--error)' }}>{crossJobSummary.failed_count}</div>
              <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>Failed</div>
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '24px', fontWeight: '600', color: 'var(--success)' }}>{crossJobSummary.status_breakdown?.RUNNING || 0}</div>
              <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>Running</div>
            </div>
          </div>

          {/* Bulk Chat Interface */}
          <div style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border-color)',
            borderRadius: '8px',
            padding: '12px',
            marginBottom: '16px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <MessageCircle size={16} style={{ color: 'var(--primary)' }} />
              <span style={{ fontSize: '14px', fontWeight: '500' }}>Ask about all jobs</span>
              <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
                ({selectedJobs.size > 0 ? `${selectedJobs.size} selected` : `${crossJobSummary.failed_count} failed`} jobs)
              </span>
            </div>

            {/* Chat Messages */}
            {bulkChatMessages.length > 0 && (
              <div style={{
                maxHeight: '200px',
                overflowY: 'auto',
                marginBottom: '8px',
                padding: '8px',
                background: 'var(--bg-tertiary)',
                borderRadius: '6px'
              }}>
                {bulkChatMessages.map((msg, idx) => (
                  <div key={idx} style={{
                    marginBottom: '8px',
                    padding: '8px',
                    background: msg.role === 'user' ? 'var(--primary-light)' : 'var(--bg-secondary)',
                    borderRadius: '6px'
                  }}>
                    <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginBottom: '4px' }}>
                      {msg.role === 'user' ? 'You' : 'AI'}
                    </div>
                    <div style={{ fontSize: '13px', whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                  </div>
                ))}
                {bulkChatLoading && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px' }}>
                    <Loader2 size={14} className="animate-spin" />
                    <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>Analyzing jobs...</span>
                  </div>
                )}
              </div>
            )}

            {/* Input */}
            <div style={{ display: 'flex', gap: '8px' }}>
              <input
                type="text"
                value={bulkChatInput}
                onChange={(e) => setBulkChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleBulkChat()}
                placeholder="Ask about outputs, ITP files, what's salvageable..."
                style={{
                  flex: 1,
                  padding: '8px 12px',
                  background: 'var(--bg-tertiary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '13px'
                }}
              />
              <button
                onClick={handleBulkChat}
                disabled={bulkChatLoading || !bulkChatInput.trim()}
                style={{
                  padding: '8px 12px',
                  background: 'var(--primary)',
                  border: 'none',
                  borderRadius: '6px',
                  color: 'white',
                  cursor: bulkChatLoading ? 'not-allowed' : 'pointer',
                  opacity: bulkChatLoading ? 0.6 : 1
                }}
              >
                <Send size={14} />
              </button>
            </div>
          </div>

          {/* Error Pattern Clusters */}
          {crossJobSummary.pattern_clusters && crossJobSummary.pattern_clusters.length > 0 && (
            <div>
              <h4 style={{ fontSize: '14px', fontWeight: '500', marginBottom: '8px', color: 'var(--text-secondary)' }}>
                Common Issues Detected
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {crossJobSummary.pattern_clusters.map((cluster, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '12px',
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div>
                        <div style={{ fontWeight: '500', marginBottom: '4px' }}>{cluster.pattern_name}</div>
                        <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>{cluster.description}</div>
                      </div>
                      <span style={{
                        background: 'var(--error-bg)',
                        color: 'var(--error)',
                        padding: '2px 8px',
                        borderRadius: '12px',
                        fontSize: '12px',
                        fontWeight: '500'
                      }}>
                        {cluster.affected_jobs} jobs
                      </span>
                    </div>

                    {cluster.suggested_fixes && cluster.suggested_fixes.length > 0 && (
                      <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid var(--border-color)' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginBottom: '4px' }}>
                          Suggested Fix (based on history):
                        </div>
                        <div style={{ fontSize: '12px', color: 'var(--success)' }}>
                          {cluster.suggested_fixes[0].modification}
                          <span style={{ marginLeft: '8px', color: 'var(--text-tertiary)' }}>
                            ({Math.round(cluster.suggested_fixes[0].success_rate * 100)}% success rate)
                          </span>
                        </div>
                      </div>
                    )}

                    <div style={{ marginTop: '8px', display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                      {cluster.job_ids.slice(0, 5).map(jid => (
                        <span key={jid} style={{
                          background: 'var(--bg-tertiary)',
                          padding: '2px 6px',
                          borderRadius: '4px',
                          fontSize: '11px',
                          fontFamily: 'var(--font-mono)'
                        }}>
                          {jid}
                        </span>
                      ))}
                      {cluster.job_ids.length > 5 && (
                        <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
                          +{cluster.job_ids.length - 5} more
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Project Notes Panel */}
      {showNotes && (
        <div style={{
          padding: '16px 24px',
          background: 'var(--bg-elevated)',
          borderBottom: '1px solid var(--border-color)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <StickyNote size={18} style={{ color: '#f97316' }} />
              <h3 style={{ fontSize: '16px', fontWeight: '600' }}>Project Notes</h3>
              <span style={{ color: 'var(--text-tertiary)', fontSize: '13px' }}>
                ({projectNotes.length} active)
              </span>
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => setShowAddNote(!showAddNote)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '6px 12px',
                  background: showAddNote ? '#f97316' : 'var(--bg-tertiary)',
                  color: showAddNote ? 'white' : 'var(--text-primary)',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                <Plus size={14} />
                Add Note
              </button>
              <button
                onClick={() => setShowNotes(false)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-tertiary)' }}
              >
                <X size={16} />
              </button>
            </div>
          </div>

          {/* Add Note Form */}
          {showAddNote && (
            <div style={{
              background: 'var(--bg-tertiary)',
              padding: '12px',
              borderRadius: '8px',
              marginBottom: '12px'
            }}>
              <textarea
                value={newNote.note}
                onChange={(e) => setNewNote({ ...newNote, note: e.target.value })}
                placeholder="Enter your note... (e.g., SURF522 failed due to quota limits)"
                style={{
                  width: '100%',
                  minHeight: '80px',
                  padding: '8px',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '13px',
                  resize: 'vertical',
                  marginBottom: '8px'
                }}
              />
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <select
                  value={newNote.category}
                  onChange={(e) => setNewNote({ ...newNote, category: e.target.value })}
                  style={{
                    padding: '6px 10px',
                    borderRadius: '6px',
                    border: '1px solid var(--border-color)',
                    background: 'var(--bg-secondary)',
                    color: 'var(--text-primary)',
                    fontSize: '12px'
                  }}
                >
                  <option value="general">General</option>
                  <option value="resource">Resource/Quota</option>
                  <option value="failure">Failure Analysis</option>
                  <option value="todo">To-Do</option>
                </select>
                <input
                  type="text"
                  value={newNote.job_names}
                  onChange={(e) => setNewNote({ ...newNote, job_names: e.target.value })}
                  placeholder="Related jobs (comma-separated, e.g., SURF522, SURF359)"
                  style={{
                    flex: 1,
                    padding: '6px 10px',
                    borderRadius: '6px',
                    border: '1px solid var(--border-color)',
                    background: 'var(--bg-secondary)',
                    color: 'var(--text-primary)',
                    fontSize: '12px'
                  }}
                />
                <button
                  onClick={handleAddNote}
                  disabled={!newNote.note.trim()}
                  style={{
                    padding: '6px 16px',
                    background: newNote.note.trim() ? '#f97316' : 'var(--bg-tertiary)',
                    color: newNote.note.trim() ? 'white' : 'var(--text-tertiary)',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: newNote.note.trim() ? 'pointer' : 'not-allowed',
                    fontSize: '12px',
                    fontWeight: '500'
                  }}
                >
                  Save
                </button>
              </div>
            </div>
          )}

          {/* Notes List */}
          {projectNotes.length === 0 ? (
            <div style={{ color: 'var(--text-tertiary)', fontSize: '13px', textAlign: 'center', padding: '20px' }}>
              No notes yet. Click "Add Note" to create one.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '300px', overflowY: 'auto' }}>
              {projectNotes.map(note => (
                <div
                  key={note.id}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '12px',
                    padding: '10px 12px',
                    background: 'var(--bg-tertiary)',
                    borderRadius: '8px',
                    borderLeft: `3px solid ${
                      note.category === 'resource' ? '#f97316' :
                      note.category === 'failure' ? 'var(--error)' :
                      note.category === 'todo' ? '#8b5cf6' :
                      'var(--border-color)'
                    }`
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '13px', color: 'var(--text-primary)', marginBottom: '4px' }}>
                      {note.note}
                    </div>
                    <div style={{ display: 'flex', gap: '8px', alignItems: 'center', fontSize: '11px', color: 'var(--text-tertiary)' }}>
                      <span style={{
                        background: note.category === 'resource' ? 'rgba(249, 115, 22, 0.15)' :
                                   note.category === 'failure' ? 'var(--error-bg)' :
                                   note.category === 'todo' ? 'rgba(139, 92, 246, 0.15)' :
                                   'var(--bg-secondary)',
                        color: note.category === 'resource' ? '#f97316' :
                               note.category === 'failure' ? 'var(--error)' :
                               note.category === 'todo' ? '#8b5cf6' :
                               'var(--text-secondary)',
                        padding: '2px 6px',
                        borderRadius: '4px',
                        fontSize: '10px',
                        textTransform: 'capitalize'
                      }}>
                        {note.category}
                      </span>
                      {note.job_names?.length > 0 && (
                        <span>Jobs: {note.job_names.join(', ')}</span>
                      )}
                      <span>{new Date(note.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '4px' }}>
                    <button
                      onClick={() => handleResolveNote(note.id)}
                      title="Mark as resolved"
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        color: 'var(--success)',
                        padding: '4px'
                      }}
                    >
                      <CheckCircle size={14} />
                    </button>
                    <button
                      onClick={() => handleDeleteNote(note.id)}
                      title="Delete note"
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        color: 'var(--error)',
                        padding: '4px'
                      }}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Batch Operation Status */}
      {batchOperationStatus && (
        <div style={{
          padding: '8px 24px',
          background: batchOperationStatus.status === 'error' ? 'var(--error-bg)' : 'var(--primary-light)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          fontSize: '13px'
        }}>
          {batchOperationStatus.status === 'running' && <Loader2 size={14} className="animate-spin" />}
          {batchOperationStatus.status === 'started' && <CheckCircle size={14} style={{ color: 'var(--success)' }} />}
          {batchOperationStatus.status === 'error' && <AlertTriangle size={14} style={{ color: 'var(--error)' }} />}
          <span>
            {batchOperationStatus.status === 'running' && `Running ${batchOperationStatus.type} on ${batchOperationStatus.count} jobs...`}
            {batchOperationStatus.status === 'started' && `${batchOperationStatus.type} started - will notify when complete`}
            {batchOperationStatus.status === 'error' && `Error: ${batchOperationStatus.message}`}
          </span>
          <button
            onClick={() => setBatchOperationStatus(null)}
            style={{ marginLeft: 'auto', background: 'none', border: 'none', color: 'inherit', cursor: 'pointer' }}
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* Active Filters */}
      {hasFilters && (
        <div style={{
          padding: '8px 24px',
          background: 'var(--bg-tertiary)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          fontSize: '13px'
        }}>
          <Filter size={14} style={{ color: 'var(--text-tertiary)' }} />
          <span style={{ color: 'var(--text-tertiary)' }}>Filters:</span>
          {statusFilter && (
            <span style={{
              padding: '2px 8px',
              background: 'var(--primary)',
              color: 'white',
              borderRadius: '4px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}>
              {statusFilter}
              <button onClick={() => setStatusFilter(null)} style={{ background: 'none', border: 'none', color: 'white', cursor: 'pointer', padding: 0 }}>
                <X size={12} />
              </button>
            </span>
          )}
          {clusterFilter && (
            <span style={{
              padding: '2px 8px',
              background: 'var(--primary)',
              color: 'white',
              borderRadius: '4px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}>
              {clusterFilter}
              <button onClick={() => setClusterFilter(null)} style={{ background: 'none', border: 'none', color: 'white', cursor: 'pointer', padding: 0 }}>
                <X size={12} />
              </button>
            </span>
          )}
          {workDirFilter && (
            <span style={{
              padding: '2px 8px',
              background: 'var(--success)',
              color: 'white',
              borderRadius: '4px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}>
              {workDirFilter}
              <button onClick={() => setWorkDirFilter(null)} style={{ background: 'none', border: 'none', color: 'white', cursor: 'pointer', padding: 0 }}>
                <X size={12} />
              </button>
            </span>
          )}
          <button
            onClick={() => { setStatusFilter(null); setClusterFilter(null); setWorkDirFilter(null); }}
            style={{
              marginLeft: 'auto',
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            Clear all
          </button>
        </div>
      )}

      {/* Main Content */}
      <main style={{ flex: 1, padding: '24px', overflowY: 'auto', marginRight: (showDiagnosisLog || showAnalytics) ? '450px' : 0, transition: 'margin-right 0.2s' }}>
        {filteredJobs.length === 0 ? (
          <EmptyState
            hasFilters={hasFilters}
            onClearFilters={() => { setStatusFilter(null); setClusterFilter(null); setWorkDirFilter(null); }}
          />
        ) : (
          <div className="jobs-grid">
            {filteredJobs.map(job => (
                <JobCard
                  key={job.job_id}
                  job={job}
                  onRemove={handleRemoveJob}
                  onDiagnose={handleDiagnose}
                  onViewDiagnosis={handleViewDiagnosis}
                  onTroubleshoot={handleTroubleshoot}
                  onMoveToNotes={openMoveToNotes}
                  selectMode={selectMode}
                  isSelected={selectedJobs.has(job.job_id)}
                  onToggleSelect={toggleJobSelection}
                />
              ))}
          </div>
        )}
      </main>

      {/* Side Panels */}
      {showDiagnosisLog && (
        <DiagnosisLog
          diagnosisLog={diagnosisLog}
          onClose={() => setShowDiagnosisLog(false)}
        />
      )}
      {showAnalytics && (
        <AnalyticsPanel
          onClose={() => setShowAnalytics(false)}
        />
      )}

      {/* Diagnosis Modal */}
      {diagnosisModal && (
        <DiagnosisModal
          job={diagnosisModal}
          onClose={() => setDiagnosisModal(null)}
          onClearDiagnosis={handleClearDiagnosis}
        />
      )}

      {/* Troubleshoot Modal */}
      {troubleshootModal && (
        <TroubleshootModal
          job={troubleshootModal}
          onClose={() => setTroubleshootModal(null)}
        />
      )}

      {/* Move to Notes Modal */}
      {moveToNotesJob && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: 'var(--bg-secondary)',
            borderRadius: '12px',
            width: '600px',
            maxHeight: '80vh',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column'
          }}>
            {/* Header */}
            <div style={{
              padding: '16px 20px',
              borderBottom: '1px solid var(--border-color)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <StickyNote size={20} style={{ color: '#f97316' }} />
                <div>
                  <div style={{ fontWeight: '600', fontSize: '16px' }}>Move to Notes</div>
                  <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
                    Job {moveToNotesJob.job_id} • {moveToNotesJob.name}
                  </div>
                </div>
              </div>
              <button
                onClick={() => setMoveToNotesJob(null)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-tertiary)' }}
              >
                <X size={20} />
              </button>
            </div>

            {/* Content */}
            <div style={{ padding: '20px', flex: 1, overflowY: 'auto' }}>
              <div style={{ marginBottom: '16px' }}>
                <label style={{ fontSize: '13px', fontWeight: '500', marginBottom: '6px', display: 'block' }}>
                  Note Content
                </label>
                <p style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginBottom: '8px' }}>
                  Edit or select parts of the AI diagnosis to include:
                </p>
                <textarea
                  value={moveToNotesText}
                  onChange={(e) => setMoveToNotesText(e.target.value)}
                  style={{
                    width: '100%',
                    minHeight: '200px',
                    padding: '12px',
                    borderRadius: '8px',
                    border: '1px solid var(--border-color)',
                    background: 'var(--bg-tertiary)',
                    color: 'var(--text-primary)',
                    fontSize: '13px',
                    resize: 'vertical',
                    fontFamily: 'inherit'
                  }}
                />
              </div>

              <div style={{ marginBottom: '16px' }}>
                <label style={{ fontSize: '13px', fontWeight: '500', marginBottom: '6px', display: 'block' }}>
                  Category
                </label>
                <select
                  value={moveToNotesCategory}
                  onChange={(e) => setMoveToNotesCategory(e.target.value)}
                  style={{
                    padding: '8px 12px',
                    borderRadius: '6px',
                    border: '1px solid var(--border-color)',
                    background: 'var(--bg-tertiary)',
                    color: 'var(--text-primary)',
                    fontSize: '13px',
                    width: '200px'
                  }}
                >
                  <option value="general">General</option>
                  <option value="resource">Resource/Quota</option>
                  <option value="failure">Failure Analysis</option>
                  <option value="todo">To-Do</option>
                </select>
              </div>

              {/* Quick extract buttons if diagnosis available */}
              {moveToNotesJob.diagnosis && (
                <div style={{ marginBottom: '16px' }}>
                  <label style={{ fontSize: '13px', fontWeight: '500', marginBottom: '6px', display: 'block' }}>
                    Quick Extract
                  </label>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    <button
                      onClick={() => {
                        const match = moveToNotesJob.diagnosis.match(/\*\*Status\*\*[:\s]*(.*?)(?=\n\n|---|\*\*)/s);
                        if (match) setMoveToNotesText(match[0].trim());
                      }}
                      style={{
                        padding: '4px 10px',
                        background: 'var(--bg-tertiary)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '4px',
                        fontSize: '11px',
                        cursor: 'pointer',
                        color: 'var(--text-secondary)'
                      }}
                    >
                      Status Only
                    </button>
                    <button
                      onClick={() => {
                        const match = moveToNotesJob.diagnosis.match(/\*\*Issues Found\*\*[\s\S]*?(?=\*\*Recommendations|---|\n\n\*\*)/);
                        if (match) setMoveToNotesText(match[0].trim());
                      }}
                      style={{
                        padding: '4px 10px',
                        background: 'var(--bg-tertiary)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '4px',
                        fontSize: '11px',
                        cursor: 'pointer',
                        color: 'var(--text-secondary)'
                      }}
                    >
                      Issues Only
                    </button>
                    <button
                      onClick={() => {
                        const match = moveToNotesJob.diagnosis.match(/\*\*Recommendations\*\*[\s\S]*/);
                        if (match) setMoveToNotesText(match[0].trim());
                      }}
                      style={{
                        padding: '4px 10px',
                        background: 'var(--bg-tertiary)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '4px',
                        fontSize: '11px',
                        cursor: 'pointer',
                        color: 'var(--text-secondary)'
                      }}
                    >
                      Recommendations Only
                    </button>
                    <button
                      onClick={() => setMoveToNotesText(moveToNotesJob.diagnosis)}
                      style={{
                        padding: '4px 10px',
                        background: 'var(--bg-tertiary)',
                        border: '1px solid var(--border-color)',
                        borderRadius: '4px',
                        fontSize: '11px',
                        cursor: 'pointer',
                        color: 'var(--text-secondary)'
                      }}
                    >
                      Full Diagnosis
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Footer */}
            <div style={{
              padding: '16px 20px',
              borderTop: '1px solid var(--border-color)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>
                This will hide the job from the queue and create a note.
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  onClick={() => setMoveToNotesJob(null)}
                  style={{
                    padding: '8px 16px',
                    background: 'var(--bg-tertiary)',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '13px',
                    color: 'var(--text-primary)'
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={handleMoveToNotes}
                  disabled={!moveToNotesText.trim()}
                  style={{
                    padding: '8px 16px',
                    background: moveToNotesText.trim() ? '#f97316' : 'var(--bg-tertiary)',
                    color: moveToNotesText.trim() ? 'white' : 'var(--text-tertiary)',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: moveToNotesText.trim() ? 'pointer' : 'not-allowed',
                    fontSize: '13px',
                    fontWeight: '500'
                  }}
                >
                  Move to Notes
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Paste Job IDs Modal */}
      {showPasteModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: 'var(--bg-secondary)',
            borderRadius: '12px',
            padding: '24px',
            width: '90%',
            maxWidth: '600px',
            maxHeight: '80vh',
            overflow: 'auto'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h2 style={{ fontSize: '18px', fontWeight: '600' }}>Paste Job IDs</h2>
              <button
                onClick={() => { setShowPasteModal(false); setPasteInput(''); setParsedJobIds([]); }}
                style={{ background: 'none', border: 'none', color: 'var(--text-tertiary)', cursor: 'pointer' }}
              >
                <X size={20} />
              </button>
            </div>

            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '12px' }}>
              Paste squeue output, "Job X: Name" format, or just job IDs. The parser will extract job IDs automatically.
            </p>

            <textarea
              value={pasteInput}
              onChange={(e) => handlePasteInputChange(e.target.value)}
              placeholder={`Examples:\n46042817 gpu-share surf-369 aaltamim R 1:23...\nJob 46042817: SURF369\n46042817, 46042818, 46042819`}
              style={{
                width: '100%',
                height: '150px',
                padding: '12px',
                background: 'var(--bg-tertiary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-mono)',
                fontSize: '12px',
                resize: 'vertical'
              }}
            />

            {parsedJobIds.length > 0 && (
              <div style={{ marginTop: '12px' }}>
                <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                  Found {parsedJobIds.length} job ID(s):
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {parsedJobIds.map(id => {
                    const exists = jobs.some(j => j.job_id === id);
                    return (
                      <span
                        key={id}
                        style={{
                          padding: '4px 8px',
                          background: exists ? 'var(--success-bg)' : 'var(--warning-bg)',
                          color: exists ? 'var(--success)' : 'var(--warning)',
                          borderRadius: '4px',
                          fontSize: '12px',
                          fontFamily: 'var(--font-mono)'
                        }}
                        title={exists ? 'Job found in dashboard' : 'Job not in dashboard'}
                      >
                        {id}
                      </span>
                    );
                  })}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginTop: '6px' }}>
                  Green = in dashboard, Yellow = not found (will be skipped)
                </div>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '16px' }}>
              <button
                onClick={() => { setShowPasteModal(false); setPasteInput(''); setParsedJobIds([]); }}
                style={{
                  padding: '8px 16px',
                  background: 'var(--bg-tertiary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button
                onClick={applyParsedJobIds}
                disabled={parsedJobIds.filter(id => jobs.some(j => j.job_id === id)).length === 0}
                style={{
                  padding: '8px 16px',
                  background: 'var(--primary)',
                  border: 'none',
                  borderRadius: '6px',
                  color: 'white',
                  cursor: 'pointer',
                  opacity: parsedJobIds.filter(id => jobs.some(j => j.job_id === id)).length === 0 ? 0.5 : 1
                }}
              >
                Select {parsedJobIds.filter(id => jobs.some(j => j.job_id === id)).length} Jobs
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer style={{
        padding: '12px 24px',
        borderTop: '1px solid var(--border-color)',
        background: 'var(--bg-secondary)',
        fontSize: '12px',
        color: 'var(--text-tertiary)',
        display: 'flex',
        justifyContent: 'space-between'
      }}>
        <span>HPC Monitor v1.1</span>
        <span>Showing {filteredJobs.length} of {jobs.length} jobs | Polling every 30s</span>
      </footer>
    </div>
  );
}

export default App;
