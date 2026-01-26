"""HPC Dashboard Backend - SSH polling service for SLURM job monitoring."""

import asyncio
import json
import subprocess
import re
import os
import sys
import tempfile
import shutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from {env_path}")
except ImportError:
    pass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sqlite3

# Add parent directory to path for slurm_debug_agent imports
# Path hierarchy: backend/ -> hpc_dashboard/ -> slurm_debug_agent/ -> MULTI-TICA/
# We need MULTI-TICA/ in path to import slurm_debug_agent
AGENT_PATH = Path(__file__).parent.parent.parent.parent
if str(AGENT_PATH) not in sys.path:
    sys.path.insert(0, str(AGENT_PATH))

# ============================================================
# Configuration
# ============================================================

@dataclass
class ClusterConfig:
    name: str
    host: str
    user: str
    ssh_key: Optional[str] = None
    scratch_path: str = "/expanse/lustre/scratch/{user}/temp_project"
    poll_interval: int = 30  # seconds


DEFAULT_CLUSTERS = {
    "euler": ClusterConfig(
        name="UW-Madison Euler",
        host="euler",  # Uses SSH config alias
        user="aaltamimi2",
        scratch_path="/srv/home/{user}",
        poll_interval=30
    ),
    "expanse": ClusterConfig(
        name="SDSC Expanse",
        host="expanse",  # Use SSH config alias for ControlMaster
        user="aaltamimi",  # Different username on expanse
        scratch_path="/expanse/lustre/scratch/{user}/temp_project",
        poll_interval=30
    )
}

# Cluster-specific SLURM configurations for troubleshoot job submission
CLUSTER_SLURM_CONFIGS = {
    "euler": {
        "account": "aaltamim",
        "partition": "gpu",
        "qos": None,  # Euler doesn't use QOS
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 8,
        "mem": "16G",
        "gpus": "1",
        "time": "48:00:00",
        "gpu_constraint": None,
    },
    "expanse": {
        "account": "wis192",
        "partition": "gpu-shared",
        "qos": "gpu-shared-normal",
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 10,
        "mem": "8G",
        "gpus": "1",
        "time": "48:00:00",
        "gpu_constraint": None,
    },
    "UW-Madison Euler": {  # Alias by display name
        "account": "aaltamim",
        "partition": "gpu",
        "qos": None,
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 8,
        "mem": "16G",
        "gpus": "1",
        "time": "48:00:00",
        "gpu_constraint": None,
    },
    "SDSC Expanse": {  # Alias by display name
        "account": "wis192",
        "partition": "gpu-shared",
        "qos": "gpu-shared-normal",
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 10,
        "mem": "8G",
        "gpus": "1",
        "time": "48:00:00",
        "gpu_constraint": None,
    },
}


def extract_slurm_directives(script_content: str) -> dict:
    """Parse #SBATCH lines from a job script to extract SLURM directives."""
    directives = {}
    for line in script_content.split('\n'):
        line = line.strip()
        if line.startswith('#SBATCH'):
            # Parse --key=value or --key value patterns
            match = re.match(r'#SBATCH\s+--(\w+[-\w]*)(?:=|\s+)(.+)', line)
            if match:
                key = match.group(1).replace('-', '_')
                value = match.group(2).strip()
                directives[key] = value
    return directives


def get_slurm_config_for_cluster(cluster_name: str, original_script: str = None) -> dict:
    """Get SLURM configuration for a cluster, optionally merging with original script directives."""
    # Start with cluster defaults
    config = CLUSTER_SLURM_CONFIGS.get(cluster_name, CLUSTER_SLURM_CONFIGS.get("expanse", {})).copy()

    # If we have an original script, extract and merge its directives
    if original_script:
        original_directives = extract_slurm_directives(original_script)
        # Prefer original script values for certain keys
        for key in ['account', 'partition', 'qos', 'time']:
            if key in original_directives:
                config[key] = original_directives[key]

    return config

# ============================================================
# Data Models
# ============================================================

@dataclass
class JobStatus:
    job_id: str
    name: str
    state: str  # RUNNING, PENDING, COMPLETED, FAILED, CANCELLED, TIMEOUT
    cluster: str
    partition: str = ""
    nodes: int = 1
    cpus: int = 1
    time_elapsed: str = "00:00:00"
    time_limit: str = "00:00:00"
    start_time: Optional[str] = None
    submit_time: Optional[str] = None
    work_dir: str = ""
    progress: float = 0.0  # 0-100
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    diagnosis: Optional[str] = None
    diagnosis_count: int = 0  # Number of times AI diagnosis has been run
    diagnosis_source: str = 'individual'  # 'individual' or 'bulk'
    diagnosis_batch_job_ids: Optional[list] = None  # List of job IDs if bulk diagnosis
    has_individual_diagnosis: bool = False  # Has been diagnosed individually
    has_bulk_diagnosis: bool = False  # Has been diagnosed via bulk
    needs_attention: bool = False
    # Job lineage fields for troubleshoot tracking
    parent_job_id: Optional[str] = None
    troubleshoot_attempt: int = 0
    modifications_applied: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class DiagnoseRequest(BaseModel):
    job_id: str
    cluster: str = "euler"
    work_dir: Optional[str] = None
    force: bool = False  # Force re-diagnosis even if one exists


class TotpRequest(BaseModel):
    code: str
    cluster: str = "expanse"


class TroubleshootRequest(BaseModel):
    job_id: str
    modifications: str
    work_dir: str
    job_name: str
    cluster: str = "expanse"
    skip_flags: list[str] = []  # e.g., ["--skip-swarmcg", "--skip-build"]


class ChatRequest(BaseModel):
    """Request for follow-up chat with AI about a job."""
    question: str
    include_logs: bool = True  # Whether to include recent logs in context


class ProjectNoteRequest(BaseModel):
    """Request to add a project note."""
    project: str = "default"
    note: str
    category: str = "general"  # 'general', 'resource', 'failure', 'todo'
    job_names: list[str] = []  # Related job names like SURF522, SURF359


# Store for chat history per job (in-memory, could be moved to DB)
job_chat_history: dict[str, list[dict]] = {}


# ============================================================
# Diagnosis Integration
# ============================================================

# Track running diagnoses to avoid duplicates
running_diagnoses: dict[str, asyncio.Task] = {}


def get_cluster_by_name(cluster_name: str) -> Optional[ClusterConfig]:
    """Get cluster config by display name or ID."""
    for cluster_id, config in DEFAULT_CLUSTERS.items():
        if config.name == cluster_name or cluster_id == cluster_name:
            return config
    return None


def scp_file(cluster: ClusterConfig, remote_path: str, local_path: Path, timeout: int = 30) -> bool:
    """Copy a file from the cluster via SCP."""
    scp_cmd = ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]

    if cluster.ssh_key:
        scp_cmd.extend(["-i", cluster.ssh_key])

    scp_cmd.append(f"{cluster.user}@{cluster.host}:{remote_path}")
    scp_cmd.append(str(local_path))

    try:
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0
    except:
        return False


async def fetch_job_files(cluster: ClusterConfig, job_id: str, work_dir: str) -> Optional[Path]:
    """Fetch job files from cluster to local temp directory."""
    # Create temp directory for this job
    temp_dir = Path(tempfile.mkdtemp(prefix=f"hpc_diag_{job_id}_"))

    # Files to fetch
    files_to_fetch = [
        f"slurm-{job_id}.out",
        f"slurm_{job_id}.out",
        "*.mdp",
        "plumed*.dat",
        "submit*.sh",
        "COLVAR",
    ]

    # Search for log files in work_dir and common subdirectories (logs/, output/)
    # Support multiple naming patterns: slurm-{job_id}.out, session_{job_id}.log, etc.
    find_patterns = [
        f'"{work_dir}" -maxdepth 1 -name "slurm*{job_id}*"',
        f'"{work_dir}" -maxdepth 1 -name "*{job_id}*.out"',
        f'"{work_dir}" -maxdepth 1 -name "*{job_id}*.log"',
        f'"{work_dir}/logs" -maxdepth 1 -name "*{job_id}*"',
        f'"{work_dir}/output" -maxdepth 1 -name "*{job_id}*"',
    ]

    find_cmd = ' '.join([f'find {p} 2>/dev/null;' for p in find_patterns])
    stdout, stderr, rc = run_ssh_command(cluster, find_cmd)

    log_files = [f.strip() for f in stdout.strip().split('\n') if f.strip()]

    if not log_files:
        # Try looking in common locations with broader search
        find_cmd = f'find "{work_dir}" -maxdepth 2 \\( -name "*.out" -o -name "*.log" \\) 2>/dev/null | head -10'
        stdout, _, _ = run_ssh_command(cluster, find_cmd)
        log_files = [f.strip() for f in stdout.strip().split('\n') if f.strip()]

    fetched_any = False

    # Fetch log files - prefer files that match the job_id
    job_specific_logs = [f for f in log_files if job_id in f]
    other_logs = [f for f in log_files if job_id not in f]

    # Prioritize job-specific logs
    for log_file in job_specific_logs + other_logs:
        local_file = temp_dir / Path(log_file).name
        if scp_file(cluster, log_file, local_file):
            fetched_any = True
            print(f"Fetched for job {job_id}: {log_file}")

    # Fetch supporting files (search up to 2 levels deep)
    for pattern in ["*.mdp", "plumed*.dat", "submit*.sh"]:
        find_cmd = f'find "{work_dir}" -maxdepth 2 -name "{pattern}" 2>/dev/null'
        stdout, _, _ = run_ssh_command(cluster, find_cmd)

        for remote_file in stdout.strip().split('\n'):
            if remote_file.strip():
                local_file = temp_dir / Path(remote_file).name
                if scp_file(cluster, remote_file, local_file):
                    print(f"Fetched: {remote_file}")

    if not fetched_any:
        shutil.rmtree(temp_dir)
        return None

    return temp_dir


def extract_structured_edits_from_diagnosis(diagnosis: str) -> list[dict]:
    """Extract actionable edits from diagnosis text.

    Returns a list of structured edits that can be applied automatically:
    [
        {"param": "MAX_SURFACTANTS", "value": "300", "action": "reduce", "confidence": 0.9},
        {"param": "BOX_SIZE", "value": "+10%", "action": "increase", "confidence": 0.7},
    ]
    """
    edits = []
    if not diagnosis:
        return edits

    diag_lower = diagnosis.lower()

    # Pattern matching for common recommendations
    patterns = [
        # Surfactant count patterns
        (r'(?:reduce|decrease|lower)\s+(?:the\s+)?(?:number\s+of\s+)?surfactant[s]?\s+(?:count\s+)?(?:to\s+)?(\d+)',
         'MAX_SURFACTANTS', 'reduce', 0.9),
        (r'surfactant\s+count\s+(?:to|of)\s+(\d+)', 'MAX_SURFACTANTS', 'set', 0.85),
        (r'(\d+)\s+surfactant[s]?', 'MAX_SURFACTANTS', 'set', 0.6),

        # Box size patterns
        (r'(?:increase|enlarge)\s+(?:the\s+)?(?:simulation\s+)?box\s+(?:size\s+)?(?:by\s+)?(\d+)%',
         'BOX_INCREASE_PERCENT', 'increase', 0.8),
        (r'larger\s+(?:simulation\s+)?box', 'BOX_INCREASE_PERCENT', 'increase', 0.7),

        # Equilibration time patterns
        (r'(?:extend|increase|longer)\s+equilibration\s+(?:time\s+)?(?:to\s+)?(\d+)\s*(?:ns|nanosecond)',
         'EQUIL_TIME_NS', 'increase', 0.8),
        (r'equilibrat(?:e|ion)\s+(?:for\s+)?(\d+)\s*(?:ns|nanosecond)', 'EQUIL_TIME_NS', 'set', 0.7),

        # Temperature patterns
        (r'(?:reduce|lower|decrease)\s+(?:the\s+)?temperature\s+(?:to\s+)?(\d+)\s*K',
         'TEMPERATURE_K', 'reduce', 0.85),
        (r'(?:increase|raise)\s+(?:the\s+)?temperature\s+(?:to\s+)?(\d+)\s*K',
         'TEMPERATURE_K', 'increase', 0.85),

        # Timestep patterns
        (r'(?:reduce|decrease|smaller)\s+(?:the\s+)?(?:time\s*)?step\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*(?:fs|femto)',
         'TIMESTEP_FS', 'reduce', 0.9),

        # LINCS warnings
        (r'LINCS\s+(?:warning|error)', 'LINCS_WARNINGS', 'address', 0.95),

        # Pressure coupling
        (r'(?:adjust|change)\s+(?:the\s+)?(?:pressure\s+)?coupling',
         'PRESSURE_COUPLING', 'adjust', 0.7),
    ]

    for pattern, param, action, base_confidence in patterns:
        matches = re.findall(pattern, diag_lower)
        if matches:
            value = matches[0] if matches[0] else None
            # Adjust confidence based on context
            confidence = base_confidence
            if 'recommend' in diag_lower or 'suggest' in diag_lower:
                confidence = min(0.95, confidence + 0.1)
            if 'try' in diag_lower or 'consider' in diag_lower:
                confidence = max(0.5, confidence - 0.1)

            edits.append({
                "param": param,
                "value": str(value) if value else None,
                "action": action,
                "confidence": round(confidence, 2),
            })

    # Deduplicate by param, keeping highest confidence
    seen = {}
    for edit in edits:
        param = edit["param"]
        if param not in seen or edit["confidence"] > seen[param]["confidence"]:
            seen[param] = edit

    return list(seen.values())


def save_diagnosis(job_id: str, diagnosis: str, tokens_used: int = 0, edits_json: str = None,
                   source: str = 'individual', batch_job_ids: list = None):
    """Save diagnosis to database with optional structured edits. Increments diagnosis_count.

    Args:
        source: 'individual' or 'bulk' to indicate how the diagnosis was triggered
        batch_job_ids: list of job IDs if this was part of a bulk diagnosis
    """
    # If no edits provided, try to extract them from the diagnosis text
    if edits_json is None:
        edits = extract_structured_edits_from_diagnosis(diagnosis)
        edits_json = json.dumps(edits) if edits else None

    conn = sqlite3.connect(DB_PATH)

    # Check if diagnosis already exists to get current count and existing flags
    cursor = conn.execute(
        "SELECT diagnosis_count, has_individual, has_bulk FROM diagnoses WHERE job_id = ?",
        (job_id,)
    )
    row = cursor.fetchone()
    new_count = (row[0] or 0) + 1 if row else 1
    existing_individual = row[1] if row else 0
    existing_bulk = row[2] if row else 0

    # Set flags based on current source, preserving existing flags
    has_individual = 1 if source == 'individual' else existing_individual
    has_bulk = 1 if source == 'bulk' else existing_bulk

    batch_ids_json = json.dumps(batch_job_ids) if batch_job_ids else None

    conn.execute(
        """INSERT OR REPLACE INTO diagnoses
           (job_id, diagnosis, timestamp, tokens_used, edits_json, diagnosis_count, source, batch_job_ids, has_individual, has_bulk)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (job_id, diagnosis, datetime.now().isoformat(), tokens_used, edits_json, new_count, source, batch_ids_json, has_individual, has_bulk)
    )
    conn.commit()
    conn.close()


def get_diagnosis_count(job_id: str) -> int:
    """Get the number of times a job has been diagnosed."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT diagnosis_count FROM diagnoses WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row and row[0] else 0


def get_diagnosis_edits(job_id: str) -> list[dict]:
    """Get structured edits from a diagnosis."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT edits_json FROM diagnoses WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()

    if row and row[0]:
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return []
    return []


def get_diagnosis(job_id: str) -> Optional[str]:
    """Get diagnosis from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT diagnosis FROM diagnoses WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


async def run_diagnosis_task(job: JobStatus, cluster: ClusterConfig,
                             source: str = 'individual', batch_job_ids: list = None) -> str:
    """Background task to run diagnosis on a job. Returns diagnosis text.

    Args:
        source: 'individual' or 'bulk' to indicate how the diagnosis was triggered
        batch_job_ids: list of job IDs if this was part of a bulk diagnosis
    """
    job_id = job.job_id
    work_dir = job.work_dir
    diagnosis = ""

    try:
        print(f"Starting diagnosis for job {job_id}")

        # Fetch files from cluster
        temp_dir = await fetch_job_files(cluster, job_id, work_dir)

        if not temp_dir:
            diagnosis = f"Could not fetch log files from {work_dir}"
            save_diagnosis(job_id, diagnosis, source=source, batch_job_ids=batch_job_ids)
            return diagnosis

        # Find log file in temp directory - prefer files matching job_id
        all_log_files = list(temp_dir.glob("slurm*.out")) + list(temp_dir.glob("*.log"))

        if not all_log_files:
            diagnosis = "No log file found after fetching files"
            save_diagnosis(job_id, diagnosis, source=source, batch_job_ids=batch_job_ids)
            shutil.rmtree(temp_dir)
            return diagnosis

        # Prefer log files that contain the job_id in their name
        job_specific = [f for f in all_log_files if job_id in f.name]
        log_path = job_specific[0] if job_specific else all_log_files[0]
        print(f"Using log file for job {job_id}: {log_path.name}")

        # Try to import and run the debug agent
        try:
            from slurm_debug_agent.config import AgentConfig
            from slurm_debug_agent.langgraph_agent import run_debug_session

            config = AgentConfig.load()

            # Explicitly set API key from environment if not already set
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            print(f"DEBUG: GEMINI_API_KEY in env: {'yes' if os.environ.get('GEMINI_API_KEY') else 'no'}")
            print(f"DEBUG: config.claude.api_key: {'set' if config.claude.api_key else 'not set'}")
            if api_key:
                config.claude.api_key = api_key
                print(f"DEBUG: Set API key from environment")

            if not config.validate_api_key():
                diagnosis = "LLM API key not configured. Set GEMINI_API_KEY environment variable."
                save_diagnosis(job_id, diagnosis, source=source, batch_job_ids=batch_job_ids)
                shutil.rmtree(temp_dir)
                return diagnosis

            # Find script
            script_path = None
            for f in temp_dir.glob("submit*.sh"):
                script_path = f
                break

            if not script_path:
                script_path = temp_dir / "submit.sh"  # Placeholder

            # Run the diagnosis
            result = await run_debug_session(
                work_dir=temp_dir,
                log_path=log_path,
                script_path=script_path,
                config=config,
                job_id=job_id,
                job_state=job.state,
                max_iterations=10,  # Limit iterations for quick diagnosis
                enable_dashboard=False,  # Don't use internal dashboard
            )

            # Extract diagnosis from result
            final_state = result.get("final_state")
            diagnosis = "Diagnosis completed but no summary generated."

            if final_state:
                for key in ["summary", "agent", "__end__"]:
                    if key in final_state:
                        messages = final_state[key].get("messages", [])
                        for msg in reversed(messages):
                            if hasattr(msg, 'content') and msg.content:
                                content = msg.content
                                if isinstance(content, list):
                                    text_parts = []
                                    for part in content:
                                        if isinstance(part, dict) and 'text' in part:
                                            text_parts.append(part['text'])
                                        elif isinstance(part, str):
                                            text_parts.append(part)
                                    content = '\n'.join(text_parts)
                                if content:
                                    diagnosis = content
                                    break
                        if diagnosis != "Diagnosis completed but no summary generated.":
                            break

            save_diagnosis(job_id, diagnosis, source=source, batch_job_ids=batch_job_ids)

            # Extract and save error patterns for clustering
            patterns = extract_error_patterns(diagnosis)
            if patterns:
                save_job_patterns(job_id, patterns)
                print(f"Extracted {len(patterns)} patterns for job {job_id}")

            # Update job in database with diagnosis
            job.diagnosis = diagnosis[:500]  # Truncate for storage
            save_job(job)

            # Broadcast update with patterns
            await manager.broadcast({
                "type": "diagnosis_complete",
                "job_id": job_id,
                "diagnosis": diagnosis,
                "patterns": patterns,
                "timestamp": datetime.now().isoformat()
            })

        except ImportError as e:
            diagnosis = f"Debug agent not available: {e}"
            save_diagnosis(job_id, diagnosis, source=source, batch_job_ids=batch_job_ids)

        finally:
            # Cleanup temp directory
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)

    except Exception as e:
        diagnosis = f"Diagnosis failed: {str(e)}"
        save_diagnosis(job_id, diagnosis, source=source, batch_job_ids=batch_job_ids)
        print(f"Diagnosis error for {job_id}: {e}")

    finally:
        # Remove from running diagnoses
        if job_id in running_diagnoses:
            del running_diagnoses[job_id]

    return diagnosis


# ============================================================
# Database
# ============================================================

DB_PATH = Path.home() / ".hpc_dashboard" / "jobs.db"

def init_db():
    """Initialize SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            cluster TEXT,
            data TEXT,
            last_seen TEXT,
            hidden INTEGER DEFAULT 0,
            parent_job_id TEXT,
            troubleshoot_attempt INTEGER DEFAULT 0,
            modifications_applied TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS diagnoses (
            job_id TEXT PRIMARY KEY,
            diagnosis TEXT,
            timestamp TEXT,
            tokens_used INTEGER DEFAULT 0,
            edits_json TEXT,
            diagnosis_count INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Error patterns table - stores extracted error signatures from diagnoses
    conn.execute("""
        CREATE TABLE IF NOT EXISTS error_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_hash TEXT UNIQUE,
            pattern_name TEXT,
            pattern_description TEXT,
            keywords TEXT,
            occurrence_count INTEGER DEFAULT 1,
            first_seen TEXT,
            last_seen TEXT
        )
    """)

    # Job-to-pattern mapping for clustering
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_patterns (
            job_id TEXT,
            pattern_hash TEXT,
            confidence REAL DEFAULT 1.0,
            extracted_at TEXT,
            PRIMARY KEY (job_id, pattern_hash)
        )
    """)

    # Fix history - track what modifications fixed what errors
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fix_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_hash TEXT,
            modification TEXT,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            last_used TEXT,
            avg_success_rate REAL DEFAULT 0.0
        )
    """)

    # Batch operations log
    conn.execute("""
        CREATE TABLE IF NOT EXISTS batch_operations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation_type TEXT,
            job_ids TEXT,
            modifications TEXT,
            status TEXT,
            created_at TEXT,
            completed_at TEXT,
            result_summary TEXT
        )
    """)

    # Add columns if they don't exist (for migration from older schema)
    try:
        conn.execute("ALTER TABLE jobs ADD COLUMN parent_job_id TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        conn.execute("ALTER TABLE jobs ADD COLUMN troubleshoot_attempt INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE jobs ADD COLUMN modifications_applied TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE diagnoses ADD COLUMN edits_json TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE diagnoses ADD COLUMN diagnosis_count INTEGER DEFAULT 1")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


def save_job(job: JobStatus, respect_hidden: bool = True):
    """Save job status to database.

    Args:
        job: The job to save
        respect_hidden: If True, don't re-add jobs that were previously hidden
    """
    conn = sqlite3.connect(DB_PATH)

    if respect_hidden:
        # Check if job is hidden - if so, don't re-add it
        cursor = conn.execute("SELECT hidden FROM jobs WHERE job_id = ?", (job.job_id,))
        row = cursor.fetchone()
        if row and row[0] == 1:
            conn.close()
            return  # Job was hidden, don't re-add

    conn.execute(
        "INSERT OR REPLACE INTO jobs (job_id, cluster, data, last_seen, hidden) VALUES (?, ?, ?, ?, 0)",
        (job.job_id, job.cluster, json.dumps(job.to_dict()), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_all_jobs() -> list[JobStatus]:
    """Get all non-hidden jobs from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT data FROM jobs WHERE hidden = 0 ORDER BY last_seen DESC")
    jobs = []
    for row in cursor.fetchall():
        data = json.loads(row[0])
        jobs.append(JobStatus(**data))
    conn.close()

    # Merge in diagnoses and diagnosis counts
    diagnoses_data = get_all_diagnoses_with_counts()
    for job in jobs:
        if job.job_id in diagnoses_data:
            job.diagnosis = diagnoses_data[job.job_id]['diagnosis']
            job.diagnosis_count = diagnoses_data[job.job_id]['count']
            job.diagnosis_source = diagnoses_data[job.job_id].get('source', 'individual')
            job.diagnosis_batch_job_ids = diagnoses_data[job.job_id].get('batch_job_ids')
            job.has_individual_diagnosis = diagnoses_data[job.job_id].get('has_individual', False)
            job.has_bulk_diagnosis = diagnoses_data[job.job_id].get('has_bulk', False)

    return jobs


def get_all_diagnoses() -> dict[str, str]:
    """Get all diagnoses as a dict keyed by job_id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT job_id, diagnosis FROM diagnoses")
    diagnoses = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return diagnoses


def get_all_diagnoses_with_counts() -> dict[str, dict]:
    """Get all diagnoses with counts, source, and batch info as a dict keyed by job_id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT job_id, diagnosis, diagnosis_count, source, batch_job_ids, has_individual, has_bulk FROM diagnoses"
    )
    diagnoses = {}
    for row in cursor.fetchall():
        batch_ids = json.loads(row[4]) if row[4] else None
        diagnoses[row[0]] = {
            'diagnosis': row[1],
            'count': row[2] or 1,
            'source': row[3] or 'individual',
            'batch_job_ids': batch_ids,
            'has_individual': bool(row[5]),
            'has_bulk': bool(row[6])
        }
    conn.close()
    return diagnoses


def hide_job(job_id: str):
    """Hide a job from the dashboard."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE jobs SET hidden = 1 WHERE job_id = ?", (job_id,))
    conn.commit()
    conn.close()


def get_hidden_job_ids() -> set[str]:
    """Get set of hidden job IDs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT job_id FROM jobs WHERE hidden = 1")
    hidden_ids = {row[0] for row in cursor.fetchall()}
    conn.close()
    return hidden_ids


def get_active_job_ids_for_cluster(cluster_name: str) -> set[str]:
    """Get job IDs that are RUNNING or PENDING for a specific cluster."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT job_id, data FROM jobs WHERE hidden = 0")
    active_ids = set()
    for row in cursor.fetchall():
        try:
            data = json.loads(row[1])
            if data.get('cluster') == cluster_name and data.get('state') in ('RUNNING', 'PENDING'):
                active_ids.add(row[0])
        except:
            pass
    conn.close()
    return active_ids


def mark_job_completed(job_id: str):
    """Mark a job as COMPLETED when it disappears from queue."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT data FROM jobs WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()
    if row:
        try:
            data = json.loads(row[0])
            data['state'] = 'COMPLETED'
            data['last_update'] = datetime.now().isoformat()
            conn.execute(
                "UPDATE jobs SET data = ? WHERE job_id = ?",
                (json.dumps(data), job_id)
            )
            conn.commit()
        except:
            pass
    conn.close()


# ============================================================
# Error Pattern Extraction & Clustering
# ============================================================

# Known error patterns with keywords for matching
KNOWN_ERROR_PATTERNS = {
    "dihedral_instability": {
        "name": "Dihedral Force Constant Instability",
        "keywords": ["negative force constant", "dihedral", "force constant", "-4.", "-5.", "Dihedral group"],
        "description": "Negative or unstable dihedral force constants causing simulation explosion"
    },
    "coordinate_corruption": {
        "name": "Coordinate File Corruption",
        "keywords": ["equi.gro", "coordinate formatting", "decimal points", "gro is fixed format", "blown-up"],
        "description": "Simulation exploded, coordinates exceeded .gro format limits"
    },
    "swarmcg_cycle2_failure": {
        "name": "SwarmCG Cycle 2 Failure",
        "keywords": ["Cycle 2", "Angles & Dihedrals", "Optimization Cycle 2", "SwarmCG"],
        "description": "SwarmCG optimization failed during angle/dihedral optimization phase"
    },
    "equilibration_failure": {
        "name": "Repeated Equilibration Failures",
        "keywords": ["Equilibration run failed", "MD run failed", "consecutive iterations"],
        "description": "Multiple equilibration or MD runs failed consecutively"
    },
    "timestep_instability": {
        "name": "Timestep Too Large",
        "keywords": ["time step", "timestep", "dt", "integration", "LINCS", "constraint"],
        "description": "Simulation timestep too large for stable integration"
    },
    "memory_error": {
        "name": "Memory/Resource Error",
        "keywords": ["out of memory", "OOM", "memory", "killed", "SIGKILL"],
        "description": "Job ran out of memory or was killed by resource manager"
    },
    "mpi_communication": {
        "name": "MPI Communication Issues",
        "keywords": ["MPI_ABORT", "OpenFabrics", "UCX", "interconnect"],
        "description": "MPI or network communication problems"
    }
}


def extract_error_patterns(diagnosis_text: str) -> list[dict]:
    """Extract error patterns from diagnosis text using keyword matching."""
    if not diagnosis_text:
        return []

    patterns_found = []
    diagnosis_lower = diagnosis_text.lower()

    for pattern_id, pattern_info in KNOWN_ERROR_PATTERNS.items():
        # Count how many keywords match
        matches = sum(1 for kw in pattern_info["keywords"] if kw.lower() in diagnosis_lower)
        if matches >= 2:  # At least 2 keywords must match
            confidence = min(1.0, matches / len(pattern_info["keywords"]))
            patterns_found.append({
                "pattern_id": pattern_id,
                "pattern_hash": hashlib.md5(pattern_id.encode()).hexdigest()[:12],
                "name": pattern_info["name"],
                "description": pattern_info["description"],
                "confidence": confidence,
                "keyword_matches": matches
            })

    # Sort by confidence
    patterns_found.sort(key=lambda x: x["confidence"], reverse=True)
    return patterns_found


def save_job_patterns(job_id: str, patterns: list[dict]):
    """Save extracted patterns for a job."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat()

    for pattern in patterns:
        # Update or insert pattern
        conn.execute("""
            INSERT INTO error_patterns (pattern_hash, pattern_name, pattern_description, keywords, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(pattern_hash) DO UPDATE SET
                occurrence_count = occurrence_count + 1,
                last_seen = ?
        """, (pattern["pattern_hash"], pattern["name"], pattern["description"],
              json.dumps(KNOWN_ERROR_PATTERNS.get(pattern["pattern_id"], {}).get("keywords", [])),
              now, now, now))

        # Link job to pattern
        conn.execute("""
            INSERT OR REPLACE INTO job_patterns (job_id, pattern_hash, confidence, extracted_at)
            VALUES (?, ?, ?, ?)
        """, (job_id, pattern["pattern_hash"], pattern["confidence"], now))

    conn.commit()
    conn.close()


def get_jobs_by_pattern(pattern_hash: str) -> list[str]:
    """Get all job IDs that have a specific error pattern."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT job_id FROM job_patterns WHERE pattern_hash = ? ORDER BY extracted_at DESC",
        (pattern_hash,)
    )
    job_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return job_ids


def get_error_clusters() -> list[dict]:
    """Get clusters of jobs grouped by error pattern."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT ep.pattern_hash, ep.pattern_name, ep.pattern_description,
               ep.occurrence_count, COUNT(jp.job_id) as job_count,
               GROUP_CONCAT(jp.job_id) as job_ids
        FROM error_patterns ep
        LEFT JOIN job_patterns jp ON ep.pattern_hash = jp.pattern_hash
        GROUP BY ep.pattern_hash
        ORDER BY job_count DESC
    """)

    clusters = []
    for row in cursor.fetchall():
        clusters.append({
            "pattern_hash": row[0],
            "pattern_name": row[1],
            "description": row[2],
            "total_occurrences": row[3],
            "current_job_count": row[4],
            "job_ids": row[5].split(",") if row[5] else []
        })

    conn.close()
    return clusters


def get_suggested_fixes(pattern_hash: str) -> list[dict]:
    """Get suggested fixes for a pattern based on history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT modification, success_count, failure_count, avg_success_rate, last_used
        FROM fix_history
        WHERE pattern_hash = ?
        ORDER BY avg_success_rate DESC, success_count DESC
        LIMIT 5
    """, (pattern_hash,))

    fixes = []
    for row in cursor.fetchall():
        fixes.append({
            "modification": row[0],
            "success_count": row[1],
            "failure_count": row[2],
            "success_rate": row[3],
            "last_used": row[4]
        })

    conn.close()
    return fixes


def record_fix_result(pattern_hash: str, modification: str, success: bool):
    """Record the result of a fix attempt for learning."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat()

    # Check if this fix exists
    cursor = conn.execute(
        "SELECT id, success_count, failure_count FROM fix_history WHERE pattern_hash = ? AND modification = ?",
        (pattern_hash, modification)
    )
    row = cursor.fetchone()

    if row:
        # Update existing
        new_success = row[1] + (1 if success else 0)
        new_failure = row[2] + (0 if success else 1)
        total = new_success + new_failure
        avg_rate = new_success / total if total > 0 else 0
        conn.execute("""
            UPDATE fix_history
            SET success_count = ?, failure_count = ?, avg_success_rate = ?, last_used = ?
            WHERE id = ?
        """, (new_success, new_failure, avg_rate, now, row[0]))
    else:
        # Insert new
        conn.execute("""
            INSERT INTO fix_history (pattern_hash, modification, success_count, failure_count, avg_success_rate, last_used)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (pattern_hash, modification, 1 if success else 0, 0 if success else 1, 1.0 if success else 0.0, now))

    conn.commit()
    conn.close()


def get_cross_job_summary(job_ids: list[str] = None) -> dict:
    """Get a summary across multiple jobs for pattern analysis."""
    conn = sqlite3.connect(DB_PATH)

    # Get all jobs or specific ones
    if job_ids:
        placeholders = ",".join("?" * len(job_ids))
        cursor = conn.execute(f"SELECT job_id, data FROM jobs WHERE job_id IN ({placeholders})", job_ids)
    else:
        cursor = conn.execute("SELECT job_id, data FROM jobs WHERE hidden = 0")

    jobs_data = []
    status_counts = {"RUNNING": 0, "PENDING": 0, "COMPLETED": 0, "FAILED": 0, "TIMEOUT": 0}

    for row in cursor.fetchall():
        try:
            data = json.loads(row[1])
            jobs_data.append(data)
            state = data.get("state", "UNKNOWN")
            if state in status_counts:
                status_counts[state] += 1
        except:
            pass

    # Get pattern clusters for failed jobs
    failed_job_ids = [j["job_id"] for j in jobs_data if j.get("state") == "FAILED"]

    pattern_summary = []
    if failed_job_ids:
        placeholders = ",".join("?" * len(failed_job_ids))
        cursor = conn.execute(f"""
            SELECT ep.pattern_name, ep.pattern_description, COUNT(jp.job_id) as count,
                   GROUP_CONCAT(jp.job_id) as job_ids, ep.pattern_hash
            FROM job_patterns jp
            JOIN error_patterns ep ON jp.pattern_hash = ep.pattern_hash
            WHERE jp.job_id IN ({placeholders})
            GROUP BY ep.pattern_hash
            ORDER BY count DESC
        """, failed_job_ids)

        for row in cursor.fetchall():
            # Get suggested fixes for this pattern
            fixes = get_suggested_fixes(row[4])
            pattern_summary.append({
                "pattern_name": row[0],
                "description": row[1],
                "affected_jobs": row[2],
                "job_ids": row[3].split(",") if row[3] else [],
                "pattern_hash": row[4],
                "suggested_fixes": fixes
            })

    conn.close()

    return {
        "total_jobs": len(jobs_data),
        "status_breakdown": status_counts,
        "failed_count": status_counts.get("FAILED", 0),
        "pattern_clusters": pattern_summary,
        "common_issue": pattern_summary[0] if pattern_summary else None
    }


# ============================================================
# SSH Commands
# ============================================================

def run_ssh_command(cluster: ClusterConfig, command: str, timeout: int = 30) -> tuple[str, str, int]:
    """Run a command on the cluster via SSH."""
    ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]

    if cluster.ssh_key:
        ssh_cmd.extend(["-i", cluster.ssh_key])

    ssh_cmd.append(f"{cluster.user}@{cluster.host}")
    ssh_cmd.append(command)

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "SSH command timed out", 1
    except Exception as e:
        return "", str(e), 1


def parse_squeue_output(output: str, cluster_name: str) -> list[JobStatus]:
    """Parse squeue output into JobStatus objects."""
    jobs = []
    lines = output.strip().split('\n')

    for line in lines[1:]:  # Skip header
        if not line.strip():
            continue

        parts = line.split('|')
        if len(parts) < 10:
            continue

        job_id, name, state, partition, nodes, cpus, time_elapsed, time_limit, start_time, submit_time = parts[:10]
        work_dir = parts[10] if len(parts) > 10 else ""

        # Determine if job needs attention
        needs_attention = state in ("FAILED", "TIMEOUT", "NODE_FAIL")

        # Clean up timestamps
        start_time_clean = start_time.strip() if start_time.strip() not in ('N/A', 'Unknown') else None
        submit_time_clean = submit_time.strip() if submit_time.strip() not in ('N/A', 'Unknown') else None

        jobs.append(JobStatus(
            job_id=job_id.strip(),
            name=name.strip()[:30],  # Truncate long names
            state=state.strip(),
            cluster=cluster_name,
            partition=partition.strip(),
            nodes=int(nodes) if nodes.strip().isdigit() else 1,
            cpus=int(cpus) if cpus.strip().isdigit() else 1,
            time_elapsed=time_elapsed.strip(),
            time_limit=time_limit.strip(),
            start_time=start_time_clean,
            submit_time=submit_time_clean,
            work_dir=work_dir.strip(),
            needs_attention=needs_attention
        ))

    return jobs


async def poll_cluster(cluster: ClusterConfig) -> list[JobStatus]:
    """Poll a cluster for job status."""
    # Get job list with squeue
    # %i=JobID, %j=Name, %T=State, %P=Partition, %D=Nodes, %C=CPUs, %M=Elapsed, %l=TimeLimit, %S=StartTime, %V=SubmitTime, %Z=WorkDir
    squeue_cmd = (
        'squeue -u $USER --format="%i|%j|%T|%P|%D|%C|%M|%l|%S|%V|%Z" --noheader'
    )

    stdout, stderr, rc = run_ssh_command(cluster, squeue_cmd)

    if rc != 0:
        print(f"SSH error polling {cluster.name}: {stderr}")
        raise Exception(f"SSH failed for {cluster.name}: {stderr[:200]}")

    jobs = parse_squeue_output(stdout, cluster.name)

    # For completed/failed jobs, check sacct (tail to get most recent)
    sacct_cmd = (
        'sacct -u $USER --starttime=$(date -d "7 days ago" +%Y-%m-%d) '
        '--format=JobID,JobName,State,Partition,NNodes,NCPUS,Elapsed,Timelimit,Start,Submit,WorkDir '
        '--parsable2 --noheader | grep -v "\\\\." | tail -50'
    )

    stdout, stderr, rc = run_ssh_command(cluster, sacct_cmd)

    if rc == 0 and stdout.strip():
        completed_jobs = parse_sacct_output(stdout, cluster.name)
        # Add completed jobs not in running list
        running_ids = {j.job_id for j in jobs}
        for job in completed_jobs:
            if job.job_id not in running_ids:
                jobs.append(job)

    return jobs


def parse_sacct_output(output: str, cluster_name: str) -> list[JobStatus]:
    """Parse sacct output into JobStatus objects."""
    jobs = []
    lines = output.strip().split('\n')

    for line in lines:
        if not line.strip():
            continue

        parts = line.split('|')
        if len(parts) < 9:
            continue

        job_id = parts[0]
        name = parts[1] if len(parts) > 1 else ""
        state = parts[2] if len(parts) > 2 else ""
        partition = parts[3] if len(parts) > 3 else ""
        nodes = parts[4] if len(parts) > 4 else "1"
        cpus = parts[5] if len(parts) > 5 else "1"
        time_elapsed = parts[6] if len(parts) > 6 else "00:00:00"
        time_limit = parts[7] if len(parts) > 7 else "00:00:00"
        start_time = parts[8] if len(parts) > 8 else None
        submit_time = parts[9] if len(parts) > 9 else None
        work_dir = parts[10] if len(parts) > 10 else ""

        # Clean up timestamps
        start_time_clean = start_time.strip() if start_time and start_time.strip() not in ('N/A', 'Unknown', '') else None
        submit_time_clean = submit_time.strip() if submit_time and submit_time.strip() not in ('N/A', 'Unknown', '') else None

        # Determine if job needs attention
        needs_attention = state in ("FAILED", "TIMEOUT", "NODE_FAIL")

        jobs.append(JobStatus(
            job_id=job_id.strip(),
            name=name.strip()[:30],
            state=state.strip(),
            cluster=cluster_name,
            partition=partition.strip(),
            nodes=int(nodes) if nodes.strip().isdigit() else 1,
            cpus=int(cpus) if cpus.strip().isdigit() else 1,
            time_elapsed=time_elapsed.strip(),
            time_limit=time_limit.strip(),
            start_time=start_time_clean,
            submit_time=submit_time_clean,
            work_dir=work_dir.strip(),
            needs_attention=needs_attention
        ))

    return jobs


# ============================================================
# Progress Estimation
# ============================================================

def estimate_progress(job: JobStatus, cluster: ClusterConfig) -> float:
    """Estimate job progress from time elapsed vs time limit."""
    try:
        def parse_time(t: str) -> int:
            """Parse SLURM time format to seconds."""
            parts = t.split(':')
            if len(parts) == 3:
                h, m, s = map(int, parts)
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = map(int, parts)
                return m * 60 + s
            return 0

        elapsed = parse_time(job.time_elapsed)
        limit = parse_time(job.time_limit)

        if limit > 0:
            return min(100.0, (elapsed / limit) * 100)
    except:
        pass

    return 0.0


# ============================================================
# WebSocket Manager
# ============================================================

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# ============================================================
# Background Polling Task
# ============================================================

polling_task = None
should_poll = True


async def polling_loop():
    """Background task that polls clusters periodically."""
    global should_poll

    while should_poll:
        for cluster_id, cluster in DEFAULT_CLUSTERS.items():
            try:
                jobs = await poll_cluster(cluster)

                # Get existing diagnoses and hidden jobs
                diagnoses = get_all_diagnoses()
                hidden_ids = get_hidden_job_ids()

                # Get current job IDs from poll
                current_job_ids = {job.job_id for job in jobs}

                # Find jobs that were RUNNING/PENDING but are no longer in queue
                # Mark them as COMPLETED
                previously_active = get_active_job_ids_for_cluster(cluster.name)
                disappeared_jobs = previously_active - current_job_ids - hidden_ids
                for job_id in disappeared_jobs:
                    mark_job_completed(job_id)
                    print(f"Job {job_id} no longer in queue - marked as COMPLETED")

                # Filter out hidden jobs and update progress
                visible_jobs = []
                for job in jobs:
                    # Skip hidden jobs entirely
                    if job.job_id in hidden_ids:
                        continue

                    job.progress = estimate_progress(job, cluster)
                    # Preserve existing diagnosis
                    if job.job_id in diagnoses:
                        job.diagnosis = diagnoses[job.job_id]
                    save_job(job)
                    visible_jobs.append(job)

                # Broadcast update with only visible jobs
                await manager.broadcast({
                    "type": "jobs_update",
                    "cluster": cluster.name,
                    "jobs": [j.to_dict() for j in visible_jobs],
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                print(f"Polling error for {cluster.name}: {e}")
                await manager.broadcast({
                    "type": "error",
                    "cluster": cluster.name,
                    "message": str(e)
                })

        await asyncio.sleep(DEFAULT_CLUSTERS["expanse"].poll_interval)


# ============================================================
# FastAPI App
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global polling_task, should_poll

    # Startup
    init_db()
    should_poll = True
    polling_task = asyncio.create_task(polling_loop())
    print("HPC Dashboard backend started")

    yield

    # Shutdown
    should_poll = False
    if polling_task:
        polling_task.cancel()
        try:
            await polling_task
        except asyncio.CancelledError:
            pass
    print("HPC Dashboard backend stopped")


app = FastAPI(
    title="HPC Dashboard",
    description="Central monitoring dashboard for HPC SLURM jobs",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/debug/env")
async def debug_env():
    """Debug endpoint to check environment variables."""
    return {
        "GEMINI_API_KEY": "set" if os.environ.get("GEMINI_API_KEY") else "not set",
        "ANTHROPIC_API_KEY": "set" if os.environ.get("ANTHROPIC_API_KEY") else "not set",
    }


@app.get("/api/clusters")
async def get_clusters():
    """Get configured clusters."""
    return {
        cluster_id: {
            "name": config.name,
            "host": config.host,
            "poll_interval": config.poll_interval
        }
        for cluster_id, config in DEFAULT_CLUSTERS.items()
    }


@app.get("/api/jobs")
async def get_jobs(cluster: Optional[str] = None):
    """Get all jobs, optionally filtered by cluster."""
    jobs = get_all_jobs()
    if cluster:
        jobs = [j for j in jobs if j.cluster == cluster]
    return {"jobs": [j.to_dict() for j in jobs]}


@app.delete("/api/jobs/{job_id}")
async def remove_job(job_id: str):
    """Hide a job from the dashboard."""
    hide_job(job_id)
    return {"status": "ok"}


def parse_time_to_hours(time_str: str) -> float:
    """Parse SLURM time format (D-HH:MM:SS or HH:MM:SS) to hours."""
    if not time_str:
        return 0.0
    try:
        if '-' in time_str:
            days, rest = time_str.split('-')
            parts = rest.split(':')
            hours = int(days) * 24 + int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            seconds = int(parts[2]) if len(parts) > 2 else 0
        else:
            parts = time_str.split(':')
            hours = int(parts[0]) if len(parts) > 0 else 0
            minutes = int(parts[1]) if len(parts) > 1 else 0
            seconds = int(parts[2]) if len(parts) > 2 else 0
        return hours + minutes / 60 + seconds / 3600
    except:
        return 0.0


@app.get("/api/analytics")
async def get_analytics():
    """Get walltime usage analytics across all jobs."""
    jobs = get_all_jobs()

    # Calculate per-cluster stats
    cluster_stats = {}
    total_hours = 0.0
    running_hours = 0.0

    for job in jobs:
        cluster = job.cluster
        if cluster not in cluster_stats:
            cluster_stats[cluster] = {
                "total_jobs": 0,
                "running_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "total_hours": 0.0,
                "running_hours": 0.0
            }

        cluster_stats[cluster]["total_jobs"] += 1
        hours = parse_time_to_hours(job.time_elapsed)

        if job.state == "RUNNING":
            cluster_stats[cluster]["running_jobs"] += 1
            cluster_stats[cluster]["running_hours"] += hours
            running_hours += hours
        elif job.state == "COMPLETED":
            cluster_stats[cluster]["completed_jobs"] += 1
        elif job.state in ("FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"):
            cluster_stats[cluster]["failed_jobs"] += 1

        cluster_stats[cluster]["total_hours"] += hours
        total_hours += hours

    # Get usage goal from database (or use default)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT value FROM settings WHERE key = 'monthly_goal_hours'")
    row = cursor.fetchone()
    monthly_goal = float(row[0]) if row else 500.0  # Default 500 hours/month
    conn.close()

    return {
        "total_hours": round(total_hours, 2),
        "running_hours": round(running_hours, 2),
        "monthly_goal": monthly_goal,
        "usage_percent": round((total_hours / monthly_goal) * 100, 1) if monthly_goal > 0 else 0,
        "cluster_stats": {
            cluster: {
                **stats,
                "total_hours": round(stats["total_hours"], 2),
                "running_hours": round(stats["running_hours"], 2)
            }
            for cluster, stats in cluster_stats.items()
        },
        "total_jobs": len(jobs)
    }


@app.post("/api/analytics/goal")
async def set_monthly_goal(goal_hours: float):
    """Set monthly walltime usage goal."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        ("monthly_goal_hours", str(goal_hours))
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "monthly_goal": goal_hours}


# Store TOTP code temporarily (expires quickly)
_totp_codes: dict[str, tuple[str, datetime]] = {}


@app.post("/api/totp")
async def submit_totp(request: TotpRequest):
    """Submit TOTP code for cluster authentication.

    Note: TOTP codes expire after ~30 seconds. This stores the code
    temporarily and attempts to use it for the next SSH connection.
    For full TOTP support, consider using SSH keys with hardware tokens
    or setting up SSH ControlMaster for persistent connections.
    """
    cluster_id = request.cluster.lower()
    _totp_codes[cluster_id] = (request.code, datetime.now())

    # Try to establish connection with TOTP
    cluster = DEFAULT_CLUSTERS.get(cluster_id)
    if not cluster:
        return {"status": "error", "message": f"Unknown cluster: {cluster_id}"}

    # Attempt a test connection (this may not work with standard SSH)
    # For now, just acknowledge receipt
    return {
        "status": "received",
        "cluster": cluster_id,
        "message": "TOTP code received. Note: Full TOTP support requires SSH key setup or ControlMaster configuration."
    }


@app.post("/api/jobs/{job_id}/diagnose")
async def diagnose_job(job_id: str, request: DiagnoseRequest, background_tasks: BackgroundTasks):
    """Trigger LLM diagnosis for a job."""
    # Check if diagnosis already running
    if job_id in running_diagnoses:
        return {
            "status": "running",
            "job_id": job_id,
            "message": "Diagnosis already in progress"
        }

    # Find job first to check state
    jobs = get_all_jobs()
    job = next((j for j in jobs if j.job_id == job_id), None)

    # For running/pending jobs, always allow re-check (force=True by default)
    force = request.force or (job and job.state in ('RUNNING', 'PENDING'))

    # Check if diagnosis already exists (skip if force or running job)
    if not force:
        existing = get_diagnosis(job_id)
        if existing:
            return {
                "status": "complete",
                "job_id": job_id,
                "diagnosis": existing,
                "message": "Diagnosis already available"
            }

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    cluster = get_cluster_by_name(job.cluster)
    if not cluster:
        cluster = get_cluster_by_name(request.cluster)

    if not cluster:
        raise HTTPException(status_code=400, detail=f"Cluster not found")

    # Use provided work_dir if available
    if request.work_dir:
        job.work_dir = request.work_dir

    if not job.work_dir:
        raise HTTPException(status_code=400, detail="Job work directory not available")

    # Start diagnosis in background
    task = asyncio.create_task(run_diagnosis_task(job, cluster))
    running_diagnoses[job_id] = task

    return {
        "status": "queued",
        "job_id": job_id,
        "message": "Diagnosis started in background"
    }


@app.get("/api/jobs/{job_id}/diagnosis")
async def get_job_diagnosis(job_id: str):
    """Get diagnosis for a job."""
    # Check if running
    if job_id in running_diagnoses:
        return {
            "status": "running",
            "job_id": job_id,
            "diagnosis": None,
            "diagnosis_count": get_diagnosis_count(job_id)
        }

    # Get from database
    diagnosis = get_diagnosis(job_id)
    diagnosis_count = get_diagnosis_count(job_id)

    if diagnosis:
        return {
            "status": "complete",
            "job_id": job_id,
            "diagnosis": diagnosis,
            "diagnosis_count": diagnosis_count
        }

    return {
        "status": "not_found",
        "job_id": job_id,
        "diagnosis": None,
        "diagnosis_count": 0
    }


@app.get("/api/jobs/{job_id}/diagnosis/edits")
async def get_diagnosis_edits_endpoint(job_id: str):
    """Get structured edits extracted from a job's diagnosis.

    Returns actionable config changes that can be applied automatically.
    """
    edits = get_diagnosis_edits(job_id)

    # Also check if diagnosis exists
    diagnosis = get_diagnosis(job_id)

    if not diagnosis:
        return {
            "status": "not_found",
            "job_id": job_id,
            "edits": [],
            "message": "No diagnosis available for this job"
        }

    return {
        "status": "complete",
        "job_id": job_id,
        "edits": edits,
        "message": f"Found {len(edits)} actionable edit(s)"
    }


@app.delete("/api/jobs/{job_id}/diagnosis")
async def delete_diagnosis(job_id: str):
    """Delete diagnosis for a job."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT job_id FROM diagnoses WHERE job_id = ?", (job_id,))
    exists = cursor.fetchone()

    if not exists:
        conn.close()
        raise HTTPException(status_code=404, detail=f"No diagnosis found for job {job_id}")

    conn.execute("DELETE FROM diagnoses WHERE job_id = ?", (job_id,))
    conn.commit()
    conn.close()

    # Also clear the job's diagnosis field in the jobs table
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE jobs SET data = json_remove(data, '$.diagnosis') WHERE job_id = ?",
        (job_id,)
    )
    conn.commit()
    conn.close()

    return {"status": "deleted", "job_id": job_id}


async def call_llm_for_chat(question: str, context: str, chat_history: list[dict]) -> str:
    """Call LLM (Gemini or Anthropic) with context for follow-up questions."""
    try:
        # Try Gemini first
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.0-flash-lite')

            # Build conversation
            messages = []
            for msg in chat_history[-10:]:  # Last 10 messages for context
                messages.append(f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}")

            history_text = "\n".join(messages) if messages else ""

            prompt = f"""You are an HPC job assistant helping analyze simulation logs and answer questions.

CONTEXT (Job Logs and Previous Diagnosis):
{context[:50000]}

{f"CONVERSATION HISTORY:{chr(10)}{history_text}" if history_text else ""}

USER QUESTION: {question}

Provide a helpful, concise answer based on the job context. If the question asks about specific frames, distances, or simulation details, search the logs for relevant information. You can suggest follow-up analysis code if helpful (Python/bash snippets that could be run on the cluster)."""

            response = model.generate_content(prompt)
            return response.text

        # Try Anthropic as fallback
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)

            messages = []
            for msg in chat_history[-10:]:
                messages.append({"role": msg['role'], "content": msg['content']})
            messages.append({"role": "user", "content": question})

            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                system=f"""You are an HPC job assistant helping analyze simulation logs and answer questions.

CONTEXT (Job Logs and Previous Diagnosis):
{context[:50000]}

Provide helpful, concise answers based on the job context. If the question asks about specific frames, distances, or simulation details, search the logs for relevant information. You can suggest follow-up analysis code if helpful.""",
                messages=messages
            )
            return response.content[0].text

        return "No LLM API key configured. Set GEMINI_API_KEY or ANTHROPIC_API_KEY."

    except Exception as e:
        return f"Error calling LLM: {str(e)}"


@app.post("/api/jobs/{job_id}/chat")
async def chat_with_job(job_id: str, request: ChatRequest):
    """Send a follow-up question about a job to the AI.

    This allows asking questions like:
    - "What distance does frame 25 correspond to?"
    - "Can you plot the density over the simulation?"
    - "What's the current sampling status?"
    """
    # Find the job
    jobs = get_all_jobs()
    job = next((j for j in jobs if j.job_id == job_id), None)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Get cluster
    cluster = get_cluster_by_name(job.cluster)
    if not cluster:
        raise HTTPException(status_code=400, detail="Cluster not found")

    # Build context from diagnosis and optionally recent logs
    context_parts = []

    # Add existing diagnosis
    diagnosis = get_diagnosis(job_id)
    if diagnosis:
        context_parts.append(f"=== PREVIOUS AI DIAGNOSIS ===\n{diagnosis}\n")

    # Add job info
    context_parts.append(f"""=== JOB INFO ===
Job ID: {job.job_id}
Name: {job.name}
State: {job.state}
Cluster: {job.cluster}
Work Dir: {job.work_dir}
Progress: {job.progress}%
Time: {job.time_elapsed} / {job.time_limit}
""")

    # Fetch recent logs if requested
    if request.include_logs and job.work_dir:
        try:
            temp_dir = await fetch_job_files(cluster, job_id, job.work_dir)
            if temp_dir:
                # Read log files
                log_files = list(temp_dir.glob("slurm*.out")) + list(temp_dir.glob("*.log")) + list(temp_dir.glob("*.err"))
                for log_file in log_files[:3]:  # Limit to 3 files
                    try:
                        content = log_file.read_text()
                        # Get last 500 lines for recent context
                        lines = content.split('\n')
                        recent = '\n'.join(lines[-500:]) if len(lines) > 500 else content
                        context_parts.append(f"=== {log_file.name} (recent) ===\n{recent}\n")
                    except:
                        pass
                shutil.rmtree(temp_dir)
        except Exception as e:
            context_parts.append(f"(Could not fetch recent logs: {e})")

    context = "\n".join(context_parts)

    # Get or initialize chat history for this job
    if job_id not in job_chat_history:
        job_chat_history[job_id] = []

    history = job_chat_history[job_id]

    # Call LLM
    response = await call_llm_for_chat(request.question, context, history)

    # Update chat history
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": response})

    # Keep only last 20 messages
    if len(history) > 20:
        job_chat_history[job_id] = history[-20:]

    return {
        "status": "success",
        "job_id": job_id,
        "question": request.question,
        "response": response,
        "history_length": len(job_chat_history[job_id])
    }


@app.delete("/api/jobs/{job_id}/chat")
async def clear_chat_history(job_id: str):
    """Clear chat history for a job."""
    if job_id in job_chat_history:
        del job_chat_history[job_id]
    return {"status": "cleared", "job_id": job_id}


# ============================================================
# Batch Operations & Cross-Job Analysis Endpoints
# ============================================================

class BulkDiagnoseRequest(BaseModel):
    job_ids: list[str]


class BatchTroubleshootRequest(BaseModel):
    job_ids: list[str]
    modifications: str
    skip_flags: list[str] = []
    cluster: str = "expanse"


@app.get("/api/patterns")
async def get_error_patterns_endpoint():
    """Get all known error patterns and their occurrence counts."""
    clusters = get_error_clusters()
    return {
        "patterns": clusters,
        "total_patterns": len(clusters)
    }


@app.get("/api/patterns/{pattern_hash}")
async def get_pattern_details(pattern_hash: str):
    """Get details about a specific error pattern including affected jobs and suggested fixes."""
    job_ids = get_jobs_by_pattern(pattern_hash)
    fixes = get_suggested_fixes(pattern_hash)

    # Get pattern info
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT pattern_name, pattern_description, occurrence_count FROM error_patterns WHERE pattern_hash = ?",
        (pattern_hash,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Pattern not found")

    return {
        "pattern_hash": pattern_hash,
        "name": row[0],
        "description": row[1],
        "total_occurrences": row[2],
        "affected_jobs": job_ids,
        "suggested_fixes": fixes
    }


@app.get("/api/summary")
async def get_cross_job_summary_endpoint():
    """Get a summary across all jobs with pattern analysis."""
    summary = get_cross_job_summary()
    return summary


@app.post("/api/summary")
async def get_filtered_summary(job_ids: list[str]):
    """Get a summary for specific jobs."""
    summary = get_cross_job_summary(job_ids)
    return summary


# ============================================================
# Project Notes API
# ============================================================

@app.get("/api/notes")
async def get_project_notes(project: str = None, include_resolved: bool = False):
    """Get all project notes, optionally filtered by project."""
    conn = sqlite3.connect(DB_PATH)
    if project:
        if include_resolved:
            cursor = conn.execute(
                "SELECT * FROM project_notes WHERE project = ? ORDER BY created_at DESC",
                (project,)
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM project_notes WHERE project = ? AND resolved = 0 ORDER BY created_at DESC",
                (project,)
            )
    else:
        if include_resolved:
            cursor = conn.execute("SELECT * FROM project_notes ORDER BY created_at DESC")
        else:
            cursor = conn.execute("SELECT * FROM project_notes WHERE resolved = 0 ORDER BY created_at DESC")

    notes = []
    for row in cursor.fetchall():
        notes.append({
            "id": row[0],
            "project": row[1],
            "note": row[2],
            "category": row[3],
            "job_names": json.loads(row[4]) if row[4] else [],
            "created_at": row[5],
            "resolved": bool(row[6])
        })
    conn.close()
    return {"notes": notes}


@app.post("/api/notes")
async def add_project_note(request: ProjectNoteRequest):
    """Add a new project note."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO project_notes (project, note, category, job_names, created_at, resolved)
           VALUES (?, ?, ?, ?, ?, 0)""",
        (request.project, request.note, request.category,
         json.dumps(request.job_names), datetime.now().isoformat())
    )
    conn.commit()
    note_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return {"status": "created", "id": note_id}


@app.put("/api/notes/{note_id}")
async def update_project_note(note_id: int, request: ProjectNoteRequest):
    """Update an existing project note."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """UPDATE project_notes SET note = ?, category = ?, job_names = ? WHERE id = ?""",
        (request.note, request.category, json.dumps(request.job_names), note_id)
    )
    conn.commit()
    conn.close()
    return {"status": "updated", "id": note_id}


@app.post("/api/notes/{note_id}/resolve")
async def resolve_project_note(note_id: int):
    """Mark a note as resolved."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE project_notes SET resolved = 1 WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()
    return {"status": "resolved", "id": note_id}


@app.delete("/api/notes/{note_id}")
async def delete_project_note(note_id: int):
    """Delete a project note."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM project_notes WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "id": note_id}


@app.post("/api/bulk-diagnose")
async def bulk_diagnose(request: BulkDiagnoseRequest, background_tasks: BackgroundTasks):
    """Start diagnosis on multiple jobs at once."""
    if not request.job_ids:
        raise HTTPException(status_code=400, detail="No job IDs provided")

    if len(request.job_ids) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 jobs per batch")

    # Start diagnoses in background
    operation_id = hashlib.md5(f"{datetime.now().isoformat()}{request.job_ids}".encode()).hexdigest()[:12]

    # Log batch operation
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO batch_operations (operation_type, job_ids, status, created_at)
        VALUES (?, ?, ?, ?)
    """, ("bulk_diagnose", json.dumps(request.job_ids), "started", datetime.now().isoformat()))
    conn.commit()
    conn.close()

    async def run_bulk_diagnoses():
        results = []
        all_jobs = get_all_jobs()  # Fetch once, not per iteration
        for job_id in request.job_ids:
            try:
                # Find job
                job = next((j for j in all_jobs if j.job_id == job_id), None)

                # Get cluster from job's cluster field
                cluster = get_cluster_by_name(job.cluster) if job else None

                if job and cluster:
                    # Call run_diagnosis_task with bulk source and sibling job IDs
                    diagnosis = await run_diagnosis_task(
                        job, cluster,
                        source='bulk',
                        batch_job_ids=request.job_ids
                    )

                    # Get pattern count for results
                    patterns = extract_error_patterns(diagnosis) if diagnosis else []

                    results.append({"job_id": job_id, "status": "completed", "patterns": len(patterns), "diagnosis_preview": diagnosis[:200] if diagnosis else "No diagnosis"})
                else:
                    results.append({"job_id": job_id, "status": "not_found"})
            except Exception as e:
                results.append({"job_id": job_id, "status": "error", "message": str(e)})

        # Update batch operation
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            UPDATE batch_operations
            SET status = ?, completed_at = ?, result_summary = ?
            WHERE operation_type = 'bulk_diagnose' AND job_ids = ?
        """, ("completed", datetime.now().isoformat(), json.dumps(results), json.dumps(request.job_ids)))
        conn.commit()
        conn.close()

        # Broadcast completion
        await manager.broadcast({
            "type": "bulk_diagnose_complete",
            "operation_id": operation_id,
            "results": results,
            "summary": get_cross_job_summary(request.job_ids)
        })

    background_tasks.add_task(run_bulk_diagnoses)

    return {
        "status": "started",
        "operation_id": operation_id,
        "job_count": len(request.job_ids),
        "message": f"Started diagnosis on {len(request.job_ids)} jobs"
    }


@app.post("/api/batch-troubleshoot")
async def batch_troubleshoot(request: BatchTroubleshootRequest, background_tasks: BackgroundTasks):
    """Apply the same troubleshooting modifications to multiple jobs."""
    if not request.job_ids:
        raise HTTPException(status_code=400, detail="No job IDs provided")

    if len(request.job_ids) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 jobs per batch troubleshoot")

    operation_id = hashlib.md5(f"{datetime.now().isoformat()}{request.job_ids}".encode()).hexdigest()[:12]

    # Log batch operation
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO batch_operations (operation_type, job_ids, modifications, status, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, ("batch_troubleshoot", json.dumps(request.job_ids), request.modifications, "started", datetime.now().isoformat()))
    conn.commit()
    conn.close()

    # Parse modifications once
    parsed_mods = parse_modifications_simple(request.modifications)

    async def run_batch_troubleshoot():
        results = []
        all_jobs = get_all_jobs()  # Fetch once, not per iteration
        for job_id in request.job_ids:
            try:
                # Find job and get cluster from job's cluster field
                job = next((j for j in all_jobs if j.job_id == job_id), None)
                cluster = get_cluster_by_name(job.cluster) if job else None

                if not job:
                    results.append({"job_id": job_id, "status": "not_found"})
                    continue

                # Create troubleshoot for this job
                # (Reuse existing troubleshoot logic)
                try:
                    # Get job's patterns for fix history tracking
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.execute("SELECT pattern_hash FROM job_patterns WHERE job_id = ?", (job_id,))
                    pattern_hashes = [row[0] for row in cursor.fetchall()]
                    conn.close()

                    # Submit troubleshoot job (simplified - actual implementation would call create_troubleshoot_job)
                    results.append({
                        "job_id": job_id,
                        "status": "queued",
                        "modifications": parsed_mods,
                        "patterns": pattern_hashes
                    })

                    # Record fix attempt for each pattern
                    for pattern_hash in pattern_hashes:
                        record_fix_result(pattern_hash, request.modifications, success=False)  # Will update to True when job completes

                except Exception as e:
                    results.append({"job_id": job_id, "status": "error", "message": str(e)})

            except Exception as e:
                results.append({"job_id": job_id, "status": "error", "message": str(e)})

        # Update batch operation
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            UPDATE batch_operations
            SET status = ?, completed_at = ?, result_summary = ?
            WHERE operation_type = 'batch_troubleshoot' AND job_ids = ?
        """, ("submitted", datetime.now().isoformat(), json.dumps(results), json.dumps(request.job_ids)))
        conn.commit()
        conn.close()

        await manager.broadcast({
            "type": "batch_troubleshoot_complete",
            "operation_id": operation_id,
            "results": results
        })

    background_tasks.add_task(run_batch_troubleshoot)

    return {
        "status": "started",
        "operation_id": operation_id,
        "job_count": len(request.job_ids),
        "modifications": parsed_mods,
        "message": f"Started batch troubleshoot for {len(request.job_ids)} jobs"
    }


@app.get("/api/batch-operations")
async def get_batch_operations():
    """Get history of batch operations."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT id, operation_type, job_ids, modifications, status, created_at, completed_at, result_summary
        FROM batch_operations
        ORDER BY created_at DESC
        LIMIT 50
    """)

    operations = []
    for row in cursor.fetchall():
        operations.append({
            "id": row[0],
            "type": row[1],
            "job_ids": json.loads(row[2]) if row[2] else [],
            "modifications": row[3],
            "status": row[4],
            "created_at": row[5],
            "completed_at": row[6],
            "result_summary": json.loads(row[7]) if row[7] else None
        })

    conn.close()
    return {"operations": operations}


@app.post("/api/fix-history/record")
async def record_fix_outcome(pattern_hash: str, modification: str, success: bool):
    """Record the outcome of a fix attempt for learning."""
    record_fix_result(pattern_hash, modification, success)
    return {"status": "recorded", "pattern_hash": pattern_hash, "success": success}


@app.get("/api/fix-history")
async def get_fix_history_endpoint():
    """Get all fix history with success rates."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT fh.pattern_hash, ep.pattern_name, fh.modification,
               fh.success_count, fh.failure_count, fh.avg_success_rate, fh.last_used
        FROM fix_history fh
        LEFT JOIN error_patterns ep ON fh.pattern_hash = ep.pattern_hash
        ORDER BY fh.avg_success_rate DESC, fh.success_count DESC
    """)

    history = []
    for row in cursor.fetchall():
        history.append({
            "pattern_hash": row[0],
            "pattern_name": row[1],
            "modification": row[2],
            "success_count": row[3],
            "failure_count": row[4],
            "success_rate": row[5],
            "last_used": row[6]
        })

    conn.close()
    return {"history": history, "total": len(history)}


# Store bulk chat history
bulk_chat_history = []


class BulkChatRequest(BaseModel):
    job_ids: list[str]
    question: str


@app.post("/api/bulk-chat")
async def bulk_chat(request: BulkChatRequest):
    """Chat about multiple jobs at once - ask questions across all their diagnoses and logs."""
    global bulk_chat_history

    if not request.job_ids:
        raise HTTPException(status_code=400, detail="No job IDs provided")

    # Gather context from all jobs
    context_parts = []
    jobs_info = []

    conn = sqlite3.connect(DB_PATH)

    for job_id in request.job_ids[:10]:  # Limit to 10 jobs
        # Get job data
        cursor = conn.execute("SELECT data FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if row:
            try:
                job_data = json.loads(row[0])
                jobs_info.append({
                    "job_id": job_id,
                    "name": job_data.get("name"),
                    "state": job_data.get("state"),
                    "cluster": job_data.get("cluster")
                })
            except:
                pass

        # Get diagnosis
        cursor = conn.execute("SELECT diagnosis FROM diagnoses WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if row and row[0]:
            context_parts.append(f"=== JOB {job_id} DIAGNOSIS ===\n{row[0]}\n")

        # Get patterns
        cursor = conn.execute("""
            SELECT ep.pattern_name, ep.pattern_description
            FROM job_patterns jp
            JOIN error_patterns ep ON jp.pattern_hash = ep.pattern_hash
            WHERE jp.job_id = ?
        """, (job_id,))
        patterns = cursor.fetchall()
        if patterns:
            pattern_text = ", ".join([f"{p[0]}" for p in patterns])
            context_parts.append(f"Detected patterns for {job_id}: {pattern_text}\n")

    conn.close()

    # Build summary header
    jobs_summary = "\n".join([f"- {j['job_id']}: {j['name']} ({j['state']}) on {j['cluster']}" for j in jobs_info])
    context = f"""JOBS BEING ANALYZED:
{jobs_summary}

DIAGNOSES AND ANALYSIS:
{chr(10).join(context_parts)}
"""

    # Build chat history context
    history_text = ""
    if bulk_chat_history:
        history_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in bulk_chat_history[-10:]
        ])

    # Call LLM
    try:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.0-flash-lite')

            prompt = f"""You are an HPC job assistant helping analyze multiple simulation jobs.

{context}

{f"CONVERSATION HISTORY:{chr(10)}{history_text}" if history_text else ""}

USER QUESTION: {request.question}

Provide a helpful answer based on ALL the jobs' diagnoses and outputs. If the user asks about specific files (like ITP files), parameters, or outputs, reference the relevant job IDs. Be specific about which jobs you're referring to.

For questions about whether outputs are usable despite failures:
- Check if the optimization made progress before failing
- Look at what iteration/stage the failure occurred
- Consider if intermediate outputs (like ITP files from earlier successful iterations) might still be valid
- Be practical - if the fits look good up to a certain point, those results may be usable"""

            response = model.generate_content(prompt)
            answer = response.text

            # Update chat history
            bulk_chat_history.append({"role": "user", "content": request.question})
            bulk_chat_history.append({"role": "assistant", "content": answer})

            # Keep only last 20 messages
            if len(bulk_chat_history) > 20:
                bulk_chat_history = bulk_chat_history[-20:]

            return {
                "status": "success",
                "question": request.question,
                "response": answer,
                "jobs_analyzed": len(jobs_info),
                "history_length": len(bulk_chat_history)
            }

        raise HTTPException(status_code=500, detail="No LLM API key configured")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.delete("/api/bulk-chat")
async def clear_bulk_chat():
    """Clear bulk chat history."""
    global bulk_chat_history
    bulk_chat_history = []
    return {"status": "cleared"}


class PreviewRequest(BaseModel):
    modifications: str
    cluster: str = "expanse"


class ConfigChange(BaseModel):
    param: str
    value: str
    action: str  # 'set', 'increase', 'reduce', 'address'
    confidence: float
    description: str = ""


def parse_modifications_simple(user_text: str) -> list[dict]:
    """Parse user modification text into structured config changes.

    This is a rule-based parser for common patterns. For more complex
    parsing, the LLM-based parser can be used.
    """
    changes = []
    text_lower = user_text.lower()

    # Surfactant patterns
    surfactant_match = re.search(r'surfactant[s]?\s+(?:count\s+)?(?:to\s+)?(\d+)', text_lower)
    if surfactant_match:
        changes.append({
            "param": "MAX_SURFACTANTS",
            "value": surfactant_match.group(1),
            "action": "set",
            "confidence": 0.95,
            "description": f"Set surfactant count to {surfactant_match.group(1)}"
        })
    elif 'reduce' in text_lower and 'surfactant' in text_lower:
        # Try to extract number
        nums = re.findall(r'(\d+)', text_lower)
        if nums:
            changes.append({
                "param": "MAX_SURFACTANTS",
                "value": nums[0],
                "action": "reduce",
                "confidence": 0.85,
                "description": f"Reduce surfactant count to {nums[0]}"
            })
        else:
            changes.append({
                "param": "MAX_SURFACTANTS",
                "value": "350",
                "action": "reduce",
                "confidence": 0.6,
                "description": "Reduce surfactant count (default: 350)"
            })

    # Box size patterns
    box_match = re.search(r'box\s+(?:size\s+)?(?:by\s+)?(\d+)%', text_lower)
    if box_match:
        changes.append({
            "param": "BOX_INCREASE_PERCENT",
            "value": box_match.group(1),
            "action": "increase",
            "confidence": 0.9,
            "description": f"Increase box size by {box_match.group(1)}%"
        })
    elif 'larger box' in text_lower or 'increase box' in text_lower or 'bigger box' in text_lower:
        changes.append({
            "param": "BOX_INCREASE_PERCENT",
            "value": "10",
            "action": "increase",
            "confidence": 0.7,
            "description": "Increase box size by 10%"
        })

    # Equilibration time patterns
    equil_match = re.search(r'equilibr(?:at|ation)\s+(?:time\s+)?(?:to\s+)?(\d+)\s*(?:ns)?', text_lower)
    if equil_match:
        changes.append({
            "param": "EQUIL_TIME_NS",
            "value": equil_match.group(1),
            "action": "set",
            "confidence": 0.9,
            "description": f"Set equilibration time to {equil_match.group(1)} ns"
        })
    elif 'longer equilibration' in text_lower or 'extend equilibration' in text_lower:
        changes.append({
            "param": "EQUIL_TIME_NS",
            "value": "5",
            "action": "increase",
            "confidence": 0.7,
            "description": "Extend equilibration time to 5 ns"
        })

    # Temperature patterns
    temp_match = re.search(r'temperature\s+(?:to\s+)?(\d+)\s*K?', text_lower)
    if temp_match:
        action = "reduce" if any(w in text_lower for w in ['reduce', 'lower', 'decrease']) else "set"
        changes.append({
            "param": "TEMPERATURE_K",
            "value": temp_match.group(1),
            "action": action,
            "confidence": 0.9,
            "description": f"{action.capitalize()} temperature to {temp_match.group(1)} K"
        })

    # Timestep patterns
    timestep_match = re.search(r'(?:time\s*)?step\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*(?:fs)?', text_lower)
    if timestep_match:
        changes.append({
            "param": "TIMESTEP_FS",
            "value": timestep_match.group(1),
            "action": "set",
            "confidence": 0.9,
            "description": f"Set timestep to {timestep_match.group(1)} fs"
        })

    return changes


@app.get("/api/jobs/{job_id}/pipeline-flags")
async def get_pipeline_flags(job_id: str):
    """Extract available resume/skip flags from the pipeline script.

    Parses the main pipeline script to find --skip-* and --resume-* flags
    so the frontend can show dynamic "Continue From" options.
    """
    jobs = get_all_jobs()
    job = next((j for j in jobs if j.job_id == job_id), None)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    cluster = get_cluster_by_name(job.cluster)
    if not cluster:
        raise HTTPException(status_code=400, detail="Cluster not found")

    work_dir = job.work_dir

    # Find and read the pipeline master script
    find_cmd = f'cat "{work_dir}/00-pipeline_master.sh" 2>/dev/null || cat "{work_dir}/submit"*.sh 2>/dev/null | head -200'
    stdout, stderr, rc = run_ssh_command(cluster, find_cmd, timeout=30)

    if not stdout.strip():
        return {
            "status": "not_found",
            "flags": [],
            "message": "Could not find pipeline script"
        }

    # Extract skip/resume flags from the script
    flags = []

    # Look for case statements or getopts that define flags
    # Pattern: --skip-STAGE) or --resume-from=STAGE
    skip_pattern = re.findall(r'--skip-(\w+)\)', stdout)
    resume_pattern = re.findall(r'--resume-from[=\s]+(\w+)', stdout)

    # Also look for flag definitions in comments or help text
    help_pattern = re.findall(r'#.*--skip-(\w+)[:\s]+([^\n]+)', stdout, re.IGNORECASE)
    help_pattern2 = re.findall(r'--skip-(\w+)\s*\)\s*\n\s*([^;]+)', stdout)

    # Build the flags list
    seen = set()
    for stage in skip_pattern:
        if stage.lower() not in seen:
            seen.add(stage.lower())
            flags.append({
                "flag": f"--skip-{stage}",
                "stage": stage,
                "type": "skip",
                "description": f"Skip {stage.replace('-', ' ').replace('_', ' ')} stage"
            })

    # Try to extract stage order from the script
    # Look for patterns like: STAGES=("build" "equil" "metad")
    stages_match = re.search(r'STAGES\s*=\s*\(([^)]+)\)', stdout)
    if stages_match:
        stages_str = stages_match.group(1)
        stages = re.findall(r'"(\w+)"', stages_str)
        for i, stage in enumerate(stages):
            if stage.lower() not in seen:
                seen.add(stage.lower())
                flags.append({
                    "flag": f"--skip-{stage}",
                    "stage": stage,
                    "type": "skip",
                    "order": i,
                    "description": f"Skip {stage.replace('-', ' ').replace('_', ' ')} stage"
                })

    # If no flags found, try a broader search
    if not flags:
        # Look for any --skip or --resume in the script
        all_flags = re.findall(r'--(skip|resume)[-_]?(\w+)', stdout, re.IGNORECASE)
        for flag_type, stage in all_flags:
            key = f"{flag_type}-{stage}".lower()
            if key not in seen:
                seen.add(key)
                flags.append({
                    "flag": f"--{flag_type}-{stage}",
                    "stage": stage,
                    "type": flag_type,
                    "description": f"{flag_type.capitalize()} {stage.replace('-', ' ')} stage"
                })

    return {
        "status": "success",
        "flags": flags,
        "script_found": True
    }


class FullPreviewRequest(BaseModel):
    modifications: str
    cluster: str = "expanse"
    skip_flags: list[str] = []


@app.post("/api/jobs/{job_id}/troubleshoot/full-preview")
async def full_preview_troubleshoot(job_id: str, request: FullPreviewRequest):
    """Generate full preview with actual file diffs.

    Returns:
    - Original config content
    - Proposed modified config content
    - Submit script that will be created
    - Parsed changes summary
    """
    jobs = get_all_jobs()
    job = next((j for j in jobs if j.job_id == job_id), None)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    cluster = get_cluster_by_name(job.cluster)
    if not cluster:
        raise HTTPException(status_code=400, detail="Cluster not found")

    work_dir = job.work_dir
    job_name = job.name
    troubleshoot_dir = f"{work_dir}/troubleshoot/{job_name}-{job_id}"

    # Fetch original config file
    config_cmd = f'cat "{work_dir}/config/pipeline_config.sh" 2>/dev/null | head -100'
    original_config, _, _ = run_ssh_command(cluster, config_cmd, timeout=30)

    # Parse modifications
    parsed_changes = parse_modifications_simple(request.modifications)

    # Generate config overrides
    config_overrides = []
    for change in parsed_changes:
        param = change['param']
        value = change['value']
        confidence = change['confidence']
        description = change.get('description', '')
        if value and confidence >= 0.5:
            config_overrides.append(f'{param}={value}  # {description} (confidence: {confidence})')
        elif param == 'BOX_INCREASE_PERCENT':
            config_overrides.append(f'# TODO: Box size increase by {value}% requires modifying base system')
        elif param == 'LINCS_WARNINGS':
            config_overrides.append('# LINCS warnings detected - consider reducing timestep')

    # Generate the proposed override config
    proposed_config = f'''# Troubleshoot config overrides for {job_name}-{job_id}
# Original job: {job_id}
# Modifications: {request.modifications}

# Source original config first
source "{work_dir}/config/pipeline_config.sh"

# Override paths to use troubleshoot directories
WORK_DIR="{troubleshoot_dir}/work"
RESULTS_DIR="{troubleshoot_dir}/results"
LOCK_DIR="{troubleshoot_dir}/lock"
LOGS_DIR="{troubleshoot_dir}/logs"

# User-requested modifications:
{chr(10).join(config_overrides) if config_overrides else "# No automatic overrides detected - edit manually if needed"}
'''

    # Get cluster SLURM config
    slurm_config = get_slurm_config_for_cluster(job.cluster)

    # Build SLURM directives
    slurm_directives = [
        f'#SBATCH --job-name={job_name}-fix',
        f'#SBATCH --account={slurm_config.get("account", "wis192")}',
        f'#SBATCH --partition={slurm_config.get("partition", "gpu-shared")}',
    ]
    if slurm_config.get("qos"):
        slurm_directives.append(f'#SBATCH --qos={slurm_config["qos"]}')
    slurm_directives.extend([
        f'#SBATCH --nodes={slurm_config.get("nodes", 1)}',
        f'#SBATCH --ntasks={slurm_config.get("ntasks", 1)}',
        f'#SBATCH --cpus-per-task={slurm_config.get("cpus_per_task", 10)}',
        f'#SBATCH --mem={slurm_config.get("mem", "8G")}',
        f'#SBATCH --gpus={slurm_config.get("gpus", "1")}',
        f'#SBATCH --time={slurm_config.get("time", "48:00:00")}',
        f'#SBATCH --output={troubleshoot_dir}/logs/troubleshoot-%j.log',
        f'#SBATCH --error={troubleshoot_dir}/logs/troubleshoot-%j.err',
    ])

    # Extract index from job name
    index_match = re.search(r'surf-?(\d+)', job_name, re.IGNORECASE)
    job_index = index_match.group(1) if index_match else job_name

    # Generate submit script
    proposed_submit = f'''#!/bin/bash
{chr(10).join(slurm_directives)}

# =============================================================================
# TROUBLESHOOTING RUN for {job_name}
# Original Job: {job_id}
# =============================================================================
# Modifications: {request.modifications}
# =============================================================================

set -euo pipefail

# Use troubleshoot directory as SCRIPT_DIR
export SLURM_SUBMIT_DIR="{troubleshoot_dir}"
SCRIPT_DIR="{troubleshoot_dir}"

# Source the troubleshoot config overrides (which sources original config first)
source "${{SCRIPT_DIR}}/config/troubleshoot_overrides.sh"

# Create required directories
mkdir -p "${{WORK_DIR}}" "${{RESULTS_DIR}}" "${{LOCK_DIR}}" "${{LOGS_DIR}}"

echo "=============================================="
echo "TROUBLESHOOTING RUN"
echo "=============================================="
echo "Original job: {job_id}"
echo "Modifications: {request.modifications}"
echo "Work dir: ${{WORK_DIR}}"
echo "Results dir: ${{RESULTS_DIR}}"
echo "Skip flags: {' '.join(request.skip_flags) if request.skip_flags else 'none'}"
echo "=============================================="

# Run the original pipeline master script
bash "{work_dir}/00-pipeline_master.sh" --index {job_index} {' '.join(request.skip_flags)} --max-iterations 1
'''

    return {
        "status": "preview",
        "job_id": job_id,
        "files": {
            "config/troubleshoot_overrides.sh": {
                "original": original_config.strip() if original_config else "# Original config not found",
                "proposed": proposed_config.strip(),
                "type": "config"
            },
            "submit_troubleshoot.sh": {
                "original": None,  # New file
                "proposed": proposed_submit.strip(),
                "type": "script"
            }
        },
        "parsed_changes": parsed_changes,
        "troubleshoot_dir": troubleshoot_dir,
        "skip_flags": request.skip_flags
    }


@app.post("/api/jobs/{job_id}/troubleshoot/preview")
async def preview_troubleshoot(job_id: str, request: PreviewRequest):
    """Preview what modifications will be applied without submitting.

    Returns parsed config changes with validation status.
    """
    # Find job
    jobs = get_all_jobs()
    job = next((j for j in jobs if j.job_id == job_id), None)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Parse modifications
    parsed_changes = parse_modifications_simple(request.modifications)

    # Get diagnosis edits for comparison
    diagnosis_edits = get_diagnosis_edits(job_id)

    # Validate parsed changes
    validation_errors = []
    validation_warnings = []

    for change in parsed_changes:
        # Check for known valid parameters
        valid_params = [
            'MAX_SURFACTANTS', 'BOX_INCREASE_PERCENT', 'EQUIL_TIME_NS',
            'TEMPERATURE_K', 'TIMESTEP_FS', 'PRESSURE_COUPLING', 'LINCS_WARNINGS'
        ]
        if change['param'] not in valid_params:
            validation_warnings.append(f"Unknown parameter: {change['param']}")

        # Check confidence
        if change['confidence'] < 0.7:
            validation_warnings.append(f"Low confidence ({change['confidence']}) for {change['param']}")

    # Check if modifications match diagnosis recommendations
    for diag_edit in diagnosis_edits:
        matching = any(c['param'] == diag_edit['param'] for c in parsed_changes)
        if not matching and diag_edit['confidence'] > 0.8:
            validation_warnings.append(
                f"Diagnosis suggests {diag_edit['action']} {diag_edit['param']} but not included"
            )

    return {
        "status": "preview",
        "job_id": job_id,
        "parsed_changes": parsed_changes,
        "diagnosis_edits": diagnosis_edits,
        "validation": {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "warnings": validation_warnings
        },
        "original_text": request.modifications
    }


@app.post("/api/jobs/{job_id}/troubleshoot")
async def troubleshoot_job(job_id: str, request: TroubleshootRequest):
    """Create a troubleshooting branch for a job and resubmit with modifications.

    This creates a troubleshoot directory that:
    1. Symlinks to original pipeline scripts (preserves SCRIPT_DIR paths)
    2. Copies config/ and creates modified pipeline_config.sh
    3. Creates separate work/results directories for this run
    4. Submits with the user's requested modifications
    """
    # Find job
    jobs = get_all_jobs()
    job = next((j for j in jobs if j.job_id == job_id), None)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Get cluster config
    cluster = get_cluster_by_name(job.cluster)
    if not cluster:
        raise HTTPException(status_code=400, detail="Cluster not found")

    work_dir = request.work_dir or job.work_dir
    job_name = request.job_name or job.name

    # Create troubleshoot directory with job_name-job_id format
    troubleshoot_dir = f"{work_dir}/troubleshoot/{job_name}-{job_id}"

    try:
        # Create troubleshoot directory structure
        mkdir_cmd = f'mkdir -p "{troubleshoot_dir}/config" "{troubleshoot_dir}/work" "{troubleshoot_dir}/results" "{troubleshoot_dir}/logs"'
        stdout, stderr, rc = run_ssh_command(cluster, mkdir_cmd)
        if rc != 0:
            return {"status": "error", "message": f"Failed to create directory: {stderr}"}

        # Create symlinks to original pipeline scripts (numbered scripts, data, topology, etc.)
        # This preserves the SCRIPT_DIR-relative paths
        symlink_cmd = f'''
            cd "{troubleshoot_dir}" && \
            for f in "{work_dir}"/[0-9]*.sh "{work_dir}"/[0-9]*.py; do
                [ -f "$f" ] && ln -sf "$f" . 2>/dev/null
            done
            [ -d "{work_dir}/data" ] && ln -sf "{work_dir}/data" . 2>/dev/null
            [ -d "{work_dir}/topology" ] && ln -sf "{work_dir}/topology" . 2>/dev/null
            [ -d "{work_dir}/automartini_outputs" ] && ln -sf "{work_dir}/automartini_outputs" . 2>/dev/null
            [ -d "{work_dir}/MDP" ] && ln -sf "{work_dir}/MDP" . 2>/dev/null
            ls -la
        '''
        stdout, stderr, rc = run_ssh_command(cluster, symlink_cmd, timeout=60)

        # Copy config directory (we'll modify pipeline_config.sh)
        copy_config_cmd = f'cp -r "{work_dir}/config/"* "{troubleshoot_dir}/config/" 2>/dev/null || true'
        run_ssh_command(cluster, copy_config_cmd, timeout=30)

        # Create troubleshoot notes file
        diagnosis = get_diagnosis(job_id) or "No diagnosis available"
        notes_content = f'''# Troubleshooting Notes for {job_name}
# Original Job ID: {job_id}
# Created: {datetime.now().isoformat()}
# Troubleshoot Directory: {troubleshoot_dir}

## User Modifications Request:
{request.modifications}

## Original Diagnosis:
{diagnosis}
'''
        escaped_notes = notes_content.replace("'", "'\\''").replace('"', '\\"')
        notes_cmd = f'''cat > "{troubleshoot_dir}/TROUBLESHOOT_NOTES.md" << 'TROUBLESHOOT_EOF'
{notes_content}
TROUBLESHOOT_EOF'''
        run_ssh_command(cluster, notes_cmd, timeout=30)

        # Create config override file that modifies specific parameters
        # Use the improved modification parser
        parsed_changes = parse_modifications_simple(request.modifications)

        # Convert parsed changes to config overrides
        config_overrides = []
        for change in parsed_changes:
            param = change['param']
            value = change['value']
            confidence = change['confidence']
            description = change.get('description', '')

            if value and confidence >= 0.5:
                config_overrides.append(f'{param}={value}  # {description} (confidence: {confidence})')
            elif param == 'BOX_INCREASE_PERCENT':
                config_overrides.append(f'# TODO: Box size increase by {value}% requires modifying base system')
            elif param == 'LINCS_WARNINGS':
                config_overrides.append('# LINCS warnings detected - consider reducing timestep or checking constraints')

        # Create the config override
        override_content = f'''# Troubleshoot config overrides for {job_name}-{job_id}
# Original job: {job_id}
# Modifications: {request.modifications}

# Source original config first
source "{work_dir}/config/pipeline_config.sh"

# Override paths to use troubleshoot directories
WORK_DIR="{troubleshoot_dir}/work"
RESULTS_DIR="{troubleshoot_dir}/results"
LOCK_DIR="{troubleshoot_dir}/lock"
LOGS_DIR="{troubleshoot_dir}/logs"

# User-requested modifications:
{chr(10).join(config_overrides) if config_overrides else "# No automatic overrides detected - edit manually if needed"}
'''
        escaped_override = override_content.replace("'", "'\\''")
        override_cmd = f'''cat > "{troubleshoot_dir}/config/troubleshoot_overrides.sh" << 'CONFIG_EOF'
{override_content}
CONFIG_EOF'''
        run_ssh_command(cluster, override_cmd, timeout=30)

        # Get cluster-specific SLURM configuration
        slurm_config = get_slurm_config_for_cluster(job.cluster)

        # Build SLURM directives dynamically
        slurm_directives = [
            f'#SBATCH --job-name={job_name}-fix',
            f'#SBATCH --account={slurm_config.get("account", "wis192")}',
            f'#SBATCH --partition={slurm_config.get("partition", "gpu-shared")}',
        ]
        if slurm_config.get("qos"):
            slurm_directives.append(f'#SBATCH --qos={slurm_config["qos"]}')
        slurm_directives.extend([
            f'#SBATCH --nodes={slurm_config.get("nodes", 1)}',
            f'#SBATCH --ntasks={slurm_config.get("ntasks", 1)}',
            f'#SBATCH --cpus-per-task={slurm_config.get("cpus_per_task", 10)}',
            f'#SBATCH --mem={slurm_config.get("mem", "8G")}',
            f'#SBATCH --gpus={slurm_config.get("gpus", "1")}',
            f'#SBATCH --time={slurm_config.get("time", "48:00:00")}',
            f'#SBATCH --output={troubleshoot_dir}/logs/troubleshoot-%j.log',
            f'#SBATCH --error={troubleshoot_dir}/logs/troubleshoot-%j.err',
        ])

        # Create submit script that uses the troubleshoot config
        submit_script = f'''#!/bin/bash
{chr(10).join(slurm_directives)}

# =============================================================================
# TROUBLESHOOTING RUN for {job_name}
# Original Job: {job_id}
# =============================================================================
# Modifications: {request.modifications}
# =============================================================================

set -euo pipefail

# Use troubleshoot directory as SCRIPT_DIR
export SLURM_SUBMIT_DIR="{troubleshoot_dir}"
SCRIPT_DIR="{troubleshoot_dir}"

# Source the troubleshoot config overrides (which sources original config first)
source "${{SCRIPT_DIR}}/config/troubleshoot_overrides.sh"

# Create required directories
mkdir -p "${{WORK_DIR}}" "${{RESULTS_DIR}}" "${{LOCK_DIR}}" "${{LOGS_DIR}}"

echo "=============================================="
echo "TROUBLESHOOTING RUN"
echo "=============================================="
echo "Original job: {job_id}"
echo "Modifications: {request.modifications}"
echo "Work dir: ${{WORK_DIR}}"
echo "Results dir: ${{RESULTS_DIR}}"
echo "MAX_SURFACTANTS: ${{MAX_SURFACTANTS:-not set}}"
echo "Skip flags: {' '.join(request.skip_flags) if request.skip_flags else 'none'}"
echo "=============================================="

# Run the original pipeline master script from this directory
# It will use the overridden config values
bash "{work_dir}/00-pipeline_master.sh" --index {job_name.replace('surf-', '').replace('SURF', '')} {' '.join(request.skip_flags) if request.skip_flags else ''} --max-iterations 1
'''
        escaped_script = submit_script.replace("'", "'\\''")
        script_cmd = f'''cat > "{troubleshoot_dir}/submit_troubleshoot.sh" << 'SUBMIT_EOF'
{submit_script}
SUBMIT_EOF
chmod +x "{troubleshoot_dir}/submit_troubleshoot.sh"'''
        run_ssh_command(cluster, script_cmd, timeout=30)

        # Submit the job
        submit_cmd = f'cd "{troubleshoot_dir}" && sbatch submit_troubleshoot.sh'
        stdout, stderr, rc = run_ssh_command(cluster, submit_cmd, timeout=30)

        if rc != 0:
            return {
                "status": "error",
                "message": f"Created troubleshoot directory but failed to submit: {stderr}",
                "troubleshoot_dir": troubleshoot_dir
            }

        # Extract job ID from sbatch output
        new_job_id = None
        if "Submitted batch job" in stdout:
            parts = stdout.strip().split()
            if len(parts) >= 4:
                new_job_id = parts[-1]

        # Count existing troubleshoot attempts for this parent job
        troubleshoot_attempt = 1
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE parent_job_id = ?",
            (job_id,)
        )
        row = cursor.fetchone()
        if row:
            troubleshoot_attempt = row[0] + 1
        conn.close()

        # Store the new job with lineage information
        if new_job_id:
            new_job = JobStatus(
                job_id=new_job_id,
                name=f"{job_name}-fix",
                state="PENDING",
                cluster=job.cluster,
                partition=slurm_config.get("partition", ""),
                work_dir=troubleshoot_dir,
            )

            # Save with lineage info
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                """INSERT OR REPLACE INTO jobs
                   (job_id, cluster, data, last_seen, hidden, parent_job_id, troubleshoot_attempt, modifications_applied)
                   VALUES (?, ?, ?, ?, 0, ?, ?, ?)""",
                (new_job_id, new_job.cluster, json.dumps(new_job.to_dict()),
                 datetime.now().isoformat(), job_id, troubleshoot_attempt,
                 json.dumps({"text": request.modifications, "parsed": parsed_changes}))
            )
            conn.commit()
            conn.close()

        return {
            "status": "success",
            "message": f"Troubleshooting job created and submitted successfully",
            "troubleshoot_dir": troubleshoot_dir,
            "new_job_id": new_job_id,
            "parent_job_id": job_id,
            "troubleshoot_attempt": troubleshoot_attempt,
            "config_overrides": config_overrides if config_overrides else ["Manual edits may be needed"],
            "parsed_changes": parsed_changes,
            "slurm_config": {k: v for k, v in slurm_config.items() if v is not None}
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "troubleshoot_dir": troubleshoot_dir
        }


@app.get("/api/jobs/{job_id}/lineage")
async def get_job_lineage(job_id: str):
    """Get the troubleshoot history/lineage for a job.

    Returns all troubleshoot attempts for a job (children) and its parent if it's a troubleshoot job.
    """
    conn = sqlite3.connect(DB_PATH)

    # Get the job itself
    cursor = conn.execute(
        "SELECT data, parent_job_id, troubleshoot_attempt, modifications_applied FROM jobs WHERE job_id = ?",
        (job_id,)
    )
    row = cursor.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_data = json.loads(row[0])
    parent_job_id = row[1]
    troubleshoot_attempt = row[2] or 0
    modifications_applied = json.loads(row[3]) if row[3] else None

    # Get all troubleshoot attempts (children) for this job
    cursor = conn.execute(
        """SELECT job_id, data, troubleshoot_attempt, modifications_applied
           FROM jobs WHERE parent_job_id = ? ORDER BY troubleshoot_attempt""",
        (job_id,)
    )
    children = []
    for child_row in cursor.fetchall():
        child_data = json.loads(child_row[1])
        children.append({
            "job_id": child_row[0],
            "name": child_data.get("name", ""),
            "state": child_data.get("state", ""),
            "attempt": child_row[2] or 0,
            "modifications": json.loads(child_row[3]) if child_row[3] else None
        })

    # Get parent job info if this is a troubleshoot job
    parent_info = None
    if parent_job_id:
        cursor = conn.execute(
            "SELECT data FROM jobs WHERE job_id = ?",
            (parent_job_id,)
        )
        parent_row = cursor.fetchone()
        if parent_row:
            parent_data = json.loads(parent_row[0])
            parent_info = {
                "job_id": parent_job_id,
                "name": parent_data.get("name", ""),
                "state": parent_data.get("state", "")
            }

    conn.close()

    return {
        "job_id": job_id,
        "is_troubleshoot_job": parent_job_id is not None,
        "parent": parent_info,
        "troubleshoot_attempt": troubleshoot_attempt,
        "modifications_applied": modifications_applied,
        "troubleshoot_history": children,
        "total_attempts": len(children)
    }


@app.post("/api/poll")
async def trigger_poll():
    """Manually trigger a poll cycle."""
    for cluster_id, cluster in DEFAULT_CLUSTERS.items():
        jobs = await poll_cluster(cluster)
        for job in jobs:
            job.progress = estimate_progress(job, cluster)
            save_job(job)
    return {"status": "ok", "jobs_count": len(get_all_jobs())}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await manager.connect(websocket)

    # Send current state on connect
    jobs = get_all_jobs()
    await websocket.send_json({
        "type": "initial",
        "jobs": [j.to_dict() for j in jobs],
        "clusters": {
            cluster_id: {"name": config.name}
            for cluster_id, config in DEFAULT_CLUSTERS.items()
        }
    })

    try:
        while True:
            # Keep connection alive, handle any client messages
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif msg.get("type") == "refresh":
                await trigger_poll()

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
