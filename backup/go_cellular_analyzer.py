#!/usr/bin/env python3
"""
Go Cellular Dependency Analyzer
===============================

A comprehensive tool for analyzing Go repositories and creating interactive
cellular dependency visualizations with draggable cells and live communication.

Features:
- Analyzes any Go repository from GitHub URL
- Identifies direct, indirect, and transitive dependencies
- Creates beautiful cellular visualization with draggable cells
- Shows real-time cell-to-cell communication
- Categorizes dependencies intelligently
- Generates interactive HTML with full features

Usage:
    python go_cellular_analyzer.py [github_repo_url]
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import argparse
import re
import math
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import time

@dataclass
class Dependency:
    """Represents a Go dependency with all its metadata"""
    name: str
    version: str
    type: str  # direct, indirect, transitive
    category: str
    import_path: str
    file_size: Optional[int] = None
    last_modified: Optional[str] = None
    description: Optional[str] = None

@dataclass
class Relationship:
    """Represents a dependency relationship between two packages"""
    source: str
    target: str
    relationship_type: str  # imports, requires, etc.
    weight: float = 1.0

class GoCellularAnalyzer:
    """Main analyzer class for Go repositories"""

    # Enhanced categorization patterns
    CATEGORIES = {
        'kubernetes': [
            'k8s.io', 'kubernetes', 'controller-runtime', 'client-go',
            'apimachinery', 'kubectl', 'kustomize', 'helm'
        ],
        'cloud': [
            'aws', 'azure', 'gcp', 'google.golang.org/api', 'cloud.google.com',
            'github.com/Azure', 'github.com/aws'
        ],
        'database': [
            'mongo', 'postgres', 'mysql', 'redis', 'cassandra', 'etcd',
            'aerospike', 'cockroach', 'influx', 'neo4j'
        ],
        'web': [
            'gin', 'echo', 'fiber', 'chi', 'mux', 'http', 'websocket',
            'grpc', 'graphql', 'rest'
        ],
        'logging': [
            'logr', 'zap', 'logrus', 'log', 'logging', 'slog'
        ],
        'testing': [
            'testify', 'ginkgo', 'gomega', 'mock', 'assert', 'testing',
            'check', 'suite'
        ],
        'crypto': [
            'crypto', 'tls', 'certificate', 'x509', 'jwt', 'oauth',
            'security', 'hash'
        ],
        'networking': [
            'grpc', 'http', 'net', 'websocket', 'tcp', 'udp', 'dns',
            'proxy', 'load'
        ],
        'data': [
            'json', 'yaml', 'toml', 'xml', 'proto', 'protobuf', 'encoding',
            'marshal', 'unmarshal', 'serialize'
        ],
        'monitoring': [
            'prometheus', 'metrics', 'tracing', 'opencensus', 'jaeger',
            'grafana', 'alert'
        ],
        'storage': [
            'storage', 'volume', 'disk', 'file', 'blob', 's3', 'minio'
        ],
        'concurrency': [
            'sync', 'context', 'goroutine', 'channel', 'worker', 'pool'
        ],
        'cli': [
            'cobra', 'cli', 'flag', 'pflag', 'command', 'terminal'
        ],
        'config': [
            'viper', 'config', 'env', 'flag', 'setting', 'option'
        ],
        'utilities': [
            'utils', 'helper', 'common', 'errors', 'time', 'string',
            'math', 'sort', 'slice'
        ],
        'external': []  # Catch-all
    }

    # Color schemes for visualization
    COLOR_SCHEMES = {
        'kubernetes': ['#326ce5', '#4285f4'],
        'cloud': ['#ff9500', '#ffb347'],
        'database': ['#4caf50', '#66bb6a'],
        'web': ['#9c27b0', '#ba68c8'],
        'logging': ['#28a745', '#20c997'],
        'testing': ['#ffc107', '#fd7e14'],
        'crypto': ['#dc3545', '#fd7e14'],
        'networking': ['#6f42c1', '#e83e8c'],
        'data': ['#17a2b8', '#20c997'],
        'monitoring': ['#ff851b', '#ffdc00'],
        'storage': ['#8b5a3c', '#795548'],
        'concurrency': ['#607d8b', '#90a4ae'],
        'cli': ['#795548', '#8d6e63'],
        'config': ['#9e9e9e', '#bdbdbd'],
        'utilities': ['#6c757d', '#adb5bd'],
        'external': ['#00d4aa', '#7209b7']
    }

    def __init__(self, repo_url: str, output_dir: str = None):
        self.repo_url = repo_url
        self.repo_name = self._extract_repo_name(repo_url)
        self.output_dir = Path(output_dir or f"./cellular_analysis_{self.repo_name}")
        self.temp_dir = None
        self.dependencies: Dict[str, Dependency] = {}
        self.relationships: List[Relationship] = []
        self.module_name = ""
        self.go_version = ""

    def _extract_repo_name(self, url: str) -> str:
        """Extract repository name from GitHub URL"""
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2:
            return f"{path_parts[-2]}_{path_parts[-1]}"
        return "unknown_repo"

    def _categorize_dependency(self, dep_name: str) -> str:
        """Categorize a dependency based on its name"""
        dep_lower = dep_name.lower()

        for category, keywords in self.CATEGORIES.items():
            if category == 'external':
                continue
            for keyword in keywords:
                if keyword.lower() in dep_lower:
                    return category
        return 'external'

    def _run_command(self, cmd: List[str], cwd: str = None) -> Tuple[bool, str]:
        """Run a shell command and return success status and output"""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0, result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def clone_repository(self) -> bool:
        """Clone the repository to a temporary directory"""
        print(f"🔄 Cloning repository: {self.repo_url}")

        self.temp_dir = tempfile.mkdtemp(prefix="go_cellular_")
        success, output = self._run_command(['git', 'clone', self.repo_url, self.temp_dir])

        if not success:
            print(f"❌ Failed to clone repository: {output}")
            return False

        # Check if go.mod exists
        go_mod_path = Path(self.temp_dir) / "go.mod"
        if not go_mod_path.exists():
            print(f"❌ No go.mod found in repository")
            return False

        print(f"✅ Repository cloned successfully")
        return True

    def analyze_go_module(self) -> bool:
        """Analyze the Go module to extract basic information"""
        print(f"📋 Analyzing Go module...")

        go_mod_path = Path(self.temp_dir) / "go.mod"
        try:
            with open(go_mod_path, 'r') as f:
                content = f.read()

            # Extract module name
            module_match = re.search(r'^module\s+(.+)$', content, re.MULTILINE)
            if module_match:
                self.module_name = module_match.group(1).strip()

            # Extract Go version
            go_match = re.search(r'^go\s+(.+)$', content, re.MULTILINE)
            if go_match:
                self.go_version = go_match.group(1).strip()

            print(f"📦 Module: {self.module_name}")
            print(f"🔧 Go Version: {self.go_version}")
            return True

        except Exception as e:
            print(f"❌ Failed to parse go.mod: {e}")
            return False

    def extract_dependencies(self) -> bool:
        """Extract all types of dependencies"""
        print(f"🔍 Extracting dependencies...")

        # 1. Get all dependencies with their types
        success, output = self._run_command([
            'go', 'list', '-m', '-f',
            '{{if not .Main}}{{.Path}} {{.Version}} {{if .Indirect}}indirect{{else}}direct{{end}}{{end}}',
            'all'
        ], cwd=self.temp_dir)

        if not success:
            print(f"⚠️ Failed to get dependency list: {output}")
            return False

        # Parse direct and indirect dependencies
        for line in output.strip().split('\n'):
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 3:
                    name, version, dep_type = parts[0], parts[1], parts[2]
                    self.dependencies[name] = Dependency(
                        name=name,
                        version=version,
                        type=dep_type,
                        category=self._categorize_dependency(name),
                        import_path=name
                    )

        print(f"✅ Found {len(self.dependencies)} direct/indirect dependencies")

        # 2. Extract transitive dependencies by building
        self._extract_transitive_dependencies()

        # 3. Get dependency relationships
        self._extract_relationships()

        return True

    def _extract_transitive_dependencies(self):
        """Extract transitive dependencies by analyzing import statements"""
        print(f"🔍 Analyzing transitive dependencies...")

        # Get all Go files and their imports
        success, output = self._run_command([
            'go', 'list', '-f', '{{.ImportPath}}: {{join .Imports " "}}', './...'
        ], cwd=self.temp_dir)

        if not success:
            print(f"⚠️ Failed to analyze package imports: {output}")
            return

        transitive_deps = set()

        for line in output.strip().split('\n'):
            if ':' in line:
                package, imports = line.split(':', 1)
                package = package.strip()
                imports = imports.strip().split()

                for imp in imports:
                    # Check if this import is not already in our dependencies
                    if imp not in self.dependencies and not imp.startswith(self.module_name):
                        # This might be a transitive dependency
                        if '.' in imp and '/' in imp:  # Looks like an external package
                            transitive_deps.add(imp)

        # Add transitive dependencies
        for dep in transitive_deps:
            if dep not in self.dependencies:
                self.dependencies[dep] = Dependency(
                    name=dep,
                    version="unknown",
                    type="transitive",
                    category=self._categorize_dependency(dep),
                    import_path=dep
                )

        print(f"✅ Found {len(transitive_deps)} additional transitive dependencies")

    def _extract_relationships(self):
        """Extract dependency relationships"""
        print(f"🔍 Extracting dependency relationships...")

        # Get module graph
        success, output = self._run_command(['go', 'mod', 'graph'], cwd=self.temp_dir)

        if success:
            for line in output.strip().split('\n'):
                if ' ' in line:
                    source, target = line.strip().split(' ', 1)
                    self.relationships.append(Relationship(
                        source=source,
                        target=target,
                        relationship_type="requires"
                    ))

        # Get package-level imports
        success, output = self._run_command([
            'go', 'list', '-f', '{{.ImportPath}}: {{join .Imports " "}}', './...'
        ], cwd=self.temp_dir)

        if success:
            for line in output.strip().split('\n'):
                if ':' in line:
                    package, imports = line.split(':', 1)
                    package = package.strip()
                    imports = imports.strip().split()

                    for imp in imports:
                        if imp in self.dependencies:
                            self.relationships.append(Relationship(
                                source=package,
                                target=imp,
                                relationship_type="imports"
                            ))

        print(f"✅ Found {len(self.relationships)} dependency relationships")

    def generate_positions(self) -> Dict[str, Dict[str, float]]:
        """Generate optimal positions for cellular layout"""
        print(f"🧮 Calculating optimal cell positions...")

        # Group dependencies by category
        categories = {}
        for dep in self.dependencies.values():
            if dep.category not in categories:
                categories[dep.category] = []
            categories[dep.category].append(dep.name)

        positions = {}
        center_x, center_y = 500, 400
        base_radius = 200

        # Position categories in a circle
        num_categories = len(categories)
        if num_categories == 0:
            return positions

        angle_step = 2 * math.pi / num_categories

        for i, (category, deps) in enumerate(categories.items()):
            base_angle = i * angle_step
            category_radius = base_radius + len(deps) * 4

            if len(deps) == 1:
                # Single dependency at category position
                x = center_x + math.cos(base_angle) * category_radius
                y = center_y + math.sin(base_angle) * category_radius
                positions[deps[0]] = {'x': x, 'y': y}
            else:
                # Multiple dependencies in a sub-circle
                sub_radius = 50 + len(deps) * 3
                sub_angle_step = 2 * math.pi / len(deps)

                for j, dep in enumerate(deps):
                    sub_angle = base_angle + (j * sub_angle_step)
                    x = (center_x + math.cos(base_angle) * category_radius +
                         math.cos(sub_angle) * sub_radius)
                    y = (center_y + math.sin(base_angle) * category_radius +
                         math.sin(sub_angle) * sub_radius)
                    positions[dep] = {'x': x, 'y': y}

        print(f"✅ Generated positions for {len(positions)} cells")
        return positions

    def create_visualization_data(self) -> Dict:
        """Create the complete data structure for visualization"""
        print(f"📊 Creating visualization data...")

        # Group dependencies by category
        categories = {}
        for category in self.CATEGORIES.keys():
            categories[category] = []

        for dep in self.dependencies.values():
            categories[dep.category].append({
                'name': dep.name,
                'version': dep.version,
                'type': dep.type,
                'import_path': dep.import_path
            })

        # Calculate statistics
        stats = {
            'total_dependencies': len(self.dependencies),
            'direct_dependencies': len([d for d in self.dependencies.values() if d.type == 'direct']),
            'indirect_dependencies': len([d for d in self.dependencies.values() if d.type == 'indirect']),
            'transitive_dependencies': len([d for d in self.dependencies.values() if d.type == 'transitive']),
            'total_relationships': len(self.relationships)
        }

        # Create visualization data
        viz_data = {
            'module_name': self.module_name,
            'go_version': self.go_version,
            'repo_url': self.repo_url,
            'dependencies': {name: asdict(dep) for name, dep in self.dependencies.items()},
            'relationships': [asdict(rel) for rel in self.relationships],
            'categories': categories,
            'stats': stats,
            'positions': self.generate_positions(),
            'color_schemes': self.COLOR_SCHEMES,
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        print(f"✅ Created visualization data with {stats['total_dependencies']} dependencies")
        return viz_data

    def generate_html_visualization(self, viz_data: Dict) -> str:
        """Generate the complete interactive HTML visualization"""
        print(f"🎨 Generating interactive HTML visualization...")

        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧬 {viz_data['module_name']} - Cellular Dependencies</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: white;
            overflow: hidden;
            user-select: none;
        }}
        
        .header {{
            padding: 15px 20px;
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            position: relative;
            z-index: 1000;
        }}
        
        .title {{
            font-size: 1.8rem;
            font-weight: bold;
            background: linear-gradient(45deg, #00f5ff, #00d4aa, #ff6b6b, #ffc107);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        
        .subtitle {{
            font-size: 0.9rem;
            opacity: 0.8;
            color: #b0b0b0;
        }}
        
        .canvas-container {{
            position: relative;
            height: calc(100vh - 80px);
            overflow: hidden;
        }}
        
        canvas {{
            width: 100%;
            height: 100%;
            display: block;
            cursor: grab;
        }}
        
        canvas.dragging {{
            cursor: grabbing;
        }}
        
        .controls {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            padding: 15px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            max-height: 500px;
            overflow-y: auto;
            width: 300px;
            z-index: 100;
        }}
        
        .control-group {{
            margin-bottom: 20px;
        }}
        
        .control-group h4 {{
            margin-bottom: 10px;
            color: #00f5ff;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .dependency-button {{
            display: block;
            width: 100%;
            padding: 8px 12px;
            margin: 3px 0;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.3s ease;
            text-align: left;
        }}
        
        .dependency-button:hover {{
            background: rgba(255, 255, 255, 0.2);
            transform: translateX(8px);
            box-shadow: 0 4px 12px rgba(0, 245, 255, 0.2);
        }}
        
        .stats {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            padding: 18px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            z-index: 100;
        }}
        
        .legend {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            padding: 18px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            max-height: 400px;
            overflow-y: auto;
            z-index: 100;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.8rem;
        }}
        
        .legend-color {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            margin-right: 10px;
            border: 2px solid rgba(255, 255, 255, 0.3);
        }}
        
        .floating-particles {{
            position: absolute;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
        }}
        
        .particle {{
            position: absolute;
            width: 2px;
            height: 2px;
            background: rgba(0, 245, 255, 0.4);
            border-radius: 50%;
            animation: float 15s infinite linear;
        }}
        
        @keyframes float {{
            from {{
                transform: translateY(100vh) translateX(0) rotate(0deg);
                opacity: 0;
            }}
            10% {{ opacity: 1; }}
            90% {{ opacity: 1; }}
            to {{
                transform: translateY(-100px) translateX(200px) rotate(360deg);
                opacity: 0;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1 class="title">🧬 {viz_data['module_name']} - Cellular Dependencies</h1>
        <p class="subtitle">
            {viz_data['stats']['total_dependencies']} dependencies • 
            Go {viz_data['go_version']} • 
            Interactive cellular ecosystem with draggable cells
        </p>
    </div>
    
    <div class="canvas-container">
        <canvas id="canvas"></canvas>
        
        <div class="floating-particles" id="particles"></div>
        
        <div class="controls">
            <div class="control-group">
                <h4>🎯 Activate Dependencies</h4>
                <div id="dependency-controls"></div>
            </div>
            
            <div class="control-group">
                <h4>🎮 Controls</h4>
                <button class="dependency-button" onclick="toggleCommunications()">
                    Toggle Communications
                </button>
                <button class="dependency-button" onclick="resetPositions()">
                    Reset Positions
                </button>
                <button class="dependency-button" onclick="toggleAnimation()">
                    Toggle Animation
                </button>
            </div>
        </div>
        
        <div class="stats">
            <div><strong>📊 {viz_data['module_name']}</strong></div>
            <div>Total Dependencies: {viz_data['stats']['total_dependencies']}</div>
            <div>Direct: {viz_data['stats']['direct_dependencies']}</div>
            <div>Indirect: {viz_data['stats']['indirect_dependencies']}</div>
            <div>Transitive: {viz_data['stats']['transitive_dependencies']}</div>
            <div>Relationships: {viz_data['stats']['total_relationships']}</div>
            <div>Go Version: {viz_data['go_version']}</div>
            <div id="active-communications">Active Communications: 0</div>
            <div id="dragging-status">Drag Status: Ready</div>
        </div>
        
        <div class="legend">
            <div><strong>🏷️ Dependency Categories</strong></div>
            <div id="legend-items"></div>
        </div>
    </div>

    <script>
        // Visualization data
        const vizData = {json.dumps(viz_data, indent=4)};
        
        class CellularDependencyVisualizer {{
            constructor() {{
                this.canvas = document.getElementById('canvas');
                this.ctx = this.canvas.getContext('2d');
                this.cells = [];
                this.communications = [];
                this.animationSpeed = 1;
                this.showCommunications = true;
                this.animationEnabled = true;
                
                // Dragging state
                this.isDragging = false;
                this.dragTarget = null;
                this.lastMousePos = {{ x: 0, y: 0 }};
                
                this.setupCanvas();
                this.initializeCells();
                this.setupControls();
                this.setupLegend();
                this.setupEventListeners();
                this.createFloatingParticles();
                this.animate();
                this.startAutoCommunications();
            }}
            
            setupCanvas() {{
                const rect = this.canvas.getBoundingClientRect();
                this.canvas.width = rect.width * 2;
                this.canvas.height = rect.height * 2;
                this.ctx.scale(2, 2);
                
                window.addEventListener('resize', () => {{
                    const rect = this.canvas.getBoundingClientRect();
                    this.canvas.width = rect.width * 2;
                    this.canvas.height = rect.height * 2;
                    this.ctx.scale(2, 2);
                }});
            }}
            
            initializeCells() {{
                const positions = vizData.positions;
                const colorSchemes = vizData.color_schemes;
                
                for (const [name, info] of Object.entries(vizData.dependencies)) {{
                    const position = positions[name] || {{ 
                        x: 300 + Math.random() * 700, 
                        y: 200 + Math.random() * 500 
                    }};
                    const colors = colorSchemes[info.category] || colorSchemes.external;
                    
                    // Determine cell size based on type
                    let baseRadius;
                    if (info.type === 'direct') {{
                        baseRadius = 25 + Math.random() * 20;
                    }} else if (info.type === 'indirect') {{
                        baseRadius = 18 + Math.random() * 12;
                    }} else {{ // transitive
                        baseRadius = 12 + Math.random() * 8;
                    }}
                    
                    const cell = {{
                        id: name,
                        name: name.split('/').pop() || name,
                        fullName: name,
                        x: position.x,
                        y: position.y,
                        originalX: position.x,
                        originalY: position.y,
                        radius: baseRadius,
                        baseRadius: baseRadius,
                        color: colors[0],
                        gradient: colors,
                        category: info.category,
                        type: info.type,
                        version: info.version,
                        activity: 0,
                        pulse: Math.random() * Math.PI * 2,
                        organelles: this.createOrganelles(info.type === 'direct' ? 6 : info.type === 'indirect' ? 4 : 2),
                        isDragging: false,
                        connections: []
                    }};
                    
                    this.cells.push(cell);
                }}
                
                console.log(`🧬 Initialized ${{this.cells.length}} cellular dependencies`);
            }}
            
            createOrganelles(count) {{
                const organelles = [];
                for (let i = 0; i < count; i++) {{
                    organelles.push({{
                        x: (Math.random() - 0.5) * 40,
                        y: (Math.random() - 0.5) * 40,
                        radius: 2 + Math.random() * 4,
                        speed: 0.3 + Math.random() * 0.8,
                        angle: Math.random() * Math.PI * 2,
                        color: `hsl(${{180 + Math.random() * 80}}, 70%, 65%)`
                    }});
                }}
                return organelles;
            }}
            
            setupEventListeners() {{
                // Mouse events for dragging
                this.canvas.addEventListener('mousedown', (e) => {{
                    const rect = this.canvas.getBoundingClientRect();
                    const mouseX = e.clientX - rect.left;
                    const mouseY = e.clientY - rect.top;
                    
                    // Find cell under mouse
                    for (let cell of this.cells) {{
                        const dx = mouseX - cell.x;
                        const dy = mouseY - cell.y;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        
                        if (distance < cell.radius) {{
                            this.isDragging = true;
                            this.dragTarget = cell;
                            cell.isDragging = true;
                            this.lastMousePos = {{ x: mouseX, y: mouseY }};
                            this.canvas.classList.add('dragging');
                            document.getElementById('dragging-status').textContent = `Dragging: ${{cell.name}}`;
                            break;
                        }}
                    }}
                }});
                
                this.canvas.addEventListener('mousemove', (e) => {{
                    if (this.isDragging && this.dragTarget) {{
                        const rect = this.canvas.getBoundingClientRect();
                        const mouseX = e.clientX - rect.left;
                        const mouseY = e.clientY - rect.top;
                        
                        // Update cell position
                        this.dragTarget.x = mouseX;
                        this.dragTarget.y = mouseY;
                        
                        // Add some activity to the dragged cell
                        this.dragTarget.activity = Math.min(1, this.dragTarget.activity + 0.1);
                        
                        this.lastMousePos = {{ x: mouseX, y: mouseY }};
                    }}
                }});
                
                this.canvas.addEventListener('mouseup', () => {{
                    if (this.isDragging && this.dragTarget) {{
                        this.dragTarget.isDragging = false;
                        this.isDragging = false;
                        this.dragTarget = null;
                        this.canvas.classList.remove('dragging');
                        document.getElementById('dragging-status').textContent = 'Drag Status: Ready';
                    }}
                }});
                
                // Touch events for mobile
                this.canvas.addEventListener('touchstart', (e) => {{
                    e.preventDefault();
                    const touch = e.touches[0];
                    const mouseEvent = new MouseEvent('mousedown', {{
                        clientX: touch.clientX,
                        clientY: touch.clientY
                    }});
                    this.canvas.dispatchEvent(mouseEvent);
                }});
                
                this.canvas.addEventListener('touchmove', (e) => {{
                    e.preventDefault();
                    const touch = e.touches[0];
                    const mouseEvent = new MouseEvent('mousemove', {{
                        clientX: touch.clientX,
                        clientY: touch.clientY
                    }});
                    this.canvas.dispatchEvent(mouseEvent);
                }});
                
                this.canvas.addEventListener('touchend', (e) => {{
                    e.preventDefault();
                    const mouseEvent = new MouseEvent('mouseup', {{}});
                    this.canvas.dispatchEvent(mouseEvent);
                }});
            }}
            
            drawCell(cell) {{
                const {{ x, y, radius, gradient, activity, pulse, organelles, type, isDragging }} = cell;
                
                // Enhanced glow for dragged cells
                const glowIntensity = isDragging ? 25 : 15;
                const glowRadius = radius + glowIntensity + Math.sin(pulse) * 5;
                const glowGradient = this.ctx.createRadialGradient(x, y, radius * 0.8, x, y, glowRadius);
                glowGradient.addColorStop(0, `${{gradient[0]}}${{isDragging ? '80' : '40'}}`);
                glowGradient.addColorStop(1, 'transparent');
                
                this.ctx.fillStyle = glowGradient;
                this.ctx.fillRect(x - glowRadius, y - glowRadius, glowRadius * 2, glowRadius * 2);
                
                // Main cell body with enhanced visuals
                const cellGradient = this.ctx.createRadialGradient(x - radius * 0.3, y - radius * 0.3, 0, x, y, radius);
                cellGradient.addColorStop(0, gradient[0] + 'DD');
                cellGradient.addColorStop(0.7, gradient[1] + 'BB');
                cellGradient.addColorStop(1, gradient[1] + '77');
                
                this.ctx.fillStyle = cellGradient;
                this.ctx.beginPath();
                this.ctx.arc(x, y, radius + Math.sin(pulse) * 3, 0, Math.PI * 2);
                this.ctx.fill();
                
                // Enhanced border for different dependency types
                let borderWidth;
                if (type === 'direct') {{
                    borderWidth = 4;
                }} else if (type === 'indirect') {{
                    borderWidth = 3;
                }} else {{ // transitive
                    borderWidth = 2;
                }}
                
                this.ctx.strokeStyle = gradient[0] + (isDragging ? 'FF' : 'CC');
                this.ctx.lineWidth = borderWidth;
                this.ctx.stroke();
                
                // Organelles with enhanced movement
                organelles.forEach(org => {{
                    org.angle += org.speed * 0.015 * this.animationSpeed;
                    const orgX = x + Math.cos(org.angle) * (radius * 0.5) + org.x * 0.4;
                    const orgY = y + Math.sin(org.angle) * (radius * 0.5) + org.y * 0.4;
                    
                    this.ctx.fillStyle = org.color + '99';
                    this.ctx.beginPath();
                    this.ctx.arc(orgX, orgY, org.radius, 0, Math.PI * 2);
                    this.ctx.fill();
                    
                    // Organelle inner glow
                    this.ctx.fillStyle = org.color + 'DD';
                    this.ctx.beginPath();
                    this.ctx.arc(orgX - 1, orgY - 1, org.radius * 0.6, 0, Math.PI * 2);
                    this.ctx.fill();
                }});
                
                // Nucleus with DNA-like structure
                const nucleusGradient = this.ctx.createRadialGradient(x, y, 0, x, y, radius * 0.3);
                nucleusGradient.addColorStop(0, '#ffffff99');
                nucleusGradient.addColorStop(0.5, gradient[0] + '77');
                nucleusGradient.addColorStop(1, gradient[1] + '55');
                
                this.ctx.fillStyle = nucleusGradient;
                this.ctx.beginPath();
                this.ctx.arc(x, y, radius * 0.25, 0, Math.PI * 2);
                this.ctx.fill();
                
                // DNA strands in nucleus
                this.ctx.strokeStyle = '#ffffff66';
                this.ctx.lineWidth = 1;
                for (let i = 0; i < 3; i++) {{
                    this.ctx.beginPath();
                    this.ctx.arc(x + Math.cos(pulse + i) * 3, y + Math.sin(pulse + i) * 3, radius * 0.1, 0, Math.PI * 2);
                    this.ctx.stroke();
                }}
                
                // Activity rings
                if (activity > 0 || isDragging) {{
                    const ringOpacity = Math.floor((activity + (isDragging ? 0.5 : 0)) * 255).toString(16).padStart(2, '0');
                    this.ctx.strokeStyle = gradient[0] + ringOpacity;
                    this.ctx.lineWidth = 3;
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, radius + 20 + activity * 15, 0, Math.PI * 2);
                    this.ctx.stroke();
                    
                    // Secondary ring
                    this.ctx.lineWidth = 2;
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, radius + 30 + activity * 20, 0, Math.PI * 2);
                    this.ctx.stroke();
                }}
                
                // Labels for larger cells
                if (radius > 20) {{
                    this.ctx.fillStyle = '#ffffff';
                    this.ctx.font = 'bold 11px Arial';
                    this.ctx.textAlign = 'center';
                    this.ctx.fillText(cell.name.substring(0, 12), x, y + radius + 18);
                    
                    // Type indicator with enhanced styling
                    let typeColor;
                    if (type === 'direct') {{
                        typeColor = '#00ff00';
                    }} else if (type === 'indirect') {{
                        typeColor = '#ffff00';
                    }} else {{
                        typeColor = '#ff8800';
                    }}
                    
                    this.ctx.fillStyle = typeColor;
                    this.ctx.font = 'bold 9px Arial';
                    this.ctx.fillText(type.toUpperCase(), x, y + radius + 32);
                }}
            }}
            
            addCommunication(fromId, toId, type = 'dependency') {{
                const fromCell = this.cells.find(cell => cell.id === fromId);
                const toCell = this.cells.find(cell => cell.id === toId);
                
                if (!fromCell || !toCell) return;
                
                const communication = {{
                    id: Date.now() + Math.random(),
                    fromCell,
                    toCell,
                    type,
                    progress: 0,
                    speed: 0.005 + Math.random() * 0.008,
                    particles: Array(3).fill().map((_, i) => ({{ 
                        id: i, 
                        offset: i * 0.2,
                        size: 3 + Math.random() * 4
                    }})),
                    color: type === 'activation' ? '#ff6b6b' : '#ffff00',
                    intensity: type === 'activation' ? 1.5 : 1.0
                }};
                
                this.communications.push(communication);
                fromCell.activity = Math.min(1, fromCell.activity + 0.4);
                toCell.activity = Math.min(1, toCell.activity + 0.3);
                
                setTimeout(() => {{
                    this.communications = this.communications.filter(c => c.id !== communication.id);
                }}, 5000);
            }}
            
            drawCommunication(comm) {{
                if (!this.showCommunications) return;
                
                const {{ fromCell, toCell, progress, particles, color, intensity }} = comm;
                const dx = toCell.x - fromCell.x;
                const dy = toCell.y - fromCell.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                // Enhanced connection line with pulsing effect
                const pulseOpacity = 0.4 + Math.sin(progress * Math.PI * 4) * 0.3;
                this.ctx.strokeStyle = `rgba(255, 255, 0, ${{pulseOpacity * intensity}})`;
                this.ctx.lineWidth = 2 * intensity;
                this.ctx.setLineDash([8, 4]);
                this.ctx.lineDashOffset = -progress * 20;
                this.ctx.beginPath();
                this.ctx.moveTo(fromCell.x, fromCell.y);
                this.ctx.lineTo(toCell.x, toCell.y);
                this.ctx.stroke();
                this.ctx.setLineDash([]);
                
                // Enhanced signal particles with trails
                particles.forEach((particle, index) => {{
                    const particleProgress = (progress + particle.offset) % 1;
                    const particleX = fromCell.x + dx * particleProgress;
                    const particleY = fromCell.y + dy * particleProgress;
                    
                    // Particle trail
                    for (let i = 0; i < 6; i++) {{
                        const trailProgress = Math.max(0, particleProgress - i * 0.08);
                        const trailX = fromCell.x + dx * trailProgress;
                        const trailY = fromCell.y + dy * trailProgress;
                        const alpha = (6 - i) / 6 * 0.8 * intensity;
                        
                        this.ctx.fillStyle = `rgba(255, 255, 0, ${{alpha}})`;
                        this.ctx.beginPath();
                        this.ctx.arc(trailX, trailY, particle.size - i * 0.5, 0, Math.PI * 2);
                        this.ctx.fill();
                    }}
                    
                    // Main particle with enhanced glow
                    const particleGradient = this.ctx.createRadialGradient(
                        particleX, particleY, 0, 
                        particleX, particleY, particle.size * 2
                    );
                    particleGradient.addColorStop(0, color);
                    particleGradient.addColorStop(0.5, color + '88');
                    particleGradient.addColorStop(1, 'transparent');
                    
                    this.ctx.fillStyle = particleGradient;
                    this.ctx.beginPath();
                    this.ctx.arc(particleX, particleY, particle.size * 2, 0, Math.PI * 2);
                    this.ctx.fill();
                    
                    // Particle core
                    this.ctx.fillStyle = '#ffffff';
                    this.ctx.beginPath();
                    this.ctx.arc(particleX, particleY, particle.size * 0.4, 0, Math.PI * 2);
                    this.ctx.fill();
                }});
            }}
            
            setupControls() {{
                const controlsContainer = document.getElementById('dependency-controls');
                
                // Group dependencies by category
                const categories = {{}};
                this.cells.forEach(cell => {{
                    if (!categories[cell.category]) {{
                        categories[cell.category] = [];
                    }}
                    categories[cell.category].push(cell);
                }});
                
                Object.entries(categories).forEach(([category, cells]) => {{
                    const categoryDiv = document.createElement('div');
                    categoryDiv.innerHTML = `<strong style="color: ${{vizData.color_schemes[category] ? vizData.color_schemes[category][0] : '#00d4aa'}};">${{category.charAt(0).toUpperCase() + category.slice(1)}} (${{cells.length}})</strong>`;
                    controlsContainer.appendChild(categoryDiv);
                    
                    cells.slice(0, 8).forEach(cell => {{
                        const button = document.createElement('button');
                        button.className = 'dependency-button';
                        button.innerHTML = `<span style="color: ${{cell.gradient[0]}};">●</span> ${{cell.name}} (${{cell.type}})`;
                        button.title = `${{cell.fullName}} - ${{cell.version}}`;
                        button.onclick = () => this.activateCell(cell.id);
                        controlsContainer.appendChild(button);
                    }});
                    
                    if (cells.length > 8) {{
                        const moreDiv = document.createElement('div');
                        moreDiv.style.fontSize = '0.7rem';
                        moreDiv.style.opacity = '0.7';
                        moreDiv.style.marginBottom = '12px';
                        moreDiv.textContent = `... and ${{cells.length - 8}} more`;
                        controlsContainer.appendChild(moreDiv);
                    }}
                }});
            }}
            
            setupLegend() {{
                const legendContainer = document.getElementById('legend-items');
                
                Object.entries(vizData.categories).forEach(([category, deps]) => {{
                    if (deps.length > 0) {{
                        const item = document.createElement('div');
                        item.className = 'legend-item';
                        
                        const color = document.createElement('div');
                        color.className = 'legend-color';
                        const colors = vizData.color_schemes[category] || vizData.color_schemes.external;
                        color.style.background = `linear-gradient(45deg, ${{colors[0]}}, ${{colors[1]}})`;
                        
                        const label = document.createElement('span');
                        label.textContent = `${{category}} (${{deps.length}})`;
                        
                        item.appendChild(color);
                        item.appendChild(label);
                        legendContainer.appendChild(item);
                    }}
                }});
            }}
            
            activateCell(cellId) {{
                const cell = this.cells.find(c => c.id === cellId);
                if (cell) {{
                    cell.activity = 1;
                    cell.pulse = Math.PI;
                    
                    // Create communications to related dependencies
                    const relationships = vizData.relationships.filter(
                        rel => rel.source === cellId || rel.target === cellId
                    );
                    
                    relationships.slice(0, 5).forEach((rel, index) => {{
                        const targetId = rel.source === cellId ? rel.target : rel.source;
                        setTimeout(() => {{
                            this.addCommunication(cellId, targetId, 'activation');
                        }}, index * 200);
                    }});
                }}
            }}
            
            startAutoCommunications() {{
                setInterval(() => {{
                    if (Math.random() < 0.25 && vizData.relationships.length > 0) {{
                        const rel = vizData.relationships[Math.floor(Math.random() * vizData.relationships.length)];
                        this.addCommunication(rel.source, rel.target, 'auto');
                    }}
                }}, 1500);
            }}
            
            createFloatingParticles() {{
                const particlesContainer = document.getElementById('particles');
                
                setInterval(() => {{
                    if (!this.animationEnabled) return;
                    
                    const particle = document.createElement('div');
                    particle.className = 'particle';
                    particle.style.left = Math.random() * 100 + '%';
                    particle.style.animationDuration = (12 + Math.random() * 8) + 's';
                    particle.style.animationDelay = Math.random() * 3 + 's';
                    particle.style.background = `hsl(${{Math.random() * 60 + 180}}, 70%, 60%)`;
                    
                    particlesContainer.appendChild(particle);
                    
                    setTimeout(() => {{
                        if (particle.parentNode) {{
                            particle.remove();
                        }}
                    }}, 20000);
                }}, 3000);
            }}
            
            animate() {{
                // Clear canvas with enhanced cosmic background
                const gradient = this.ctx.createLinearGradient(0, 0, this.canvas.width / 2, this.canvas.height / 2);
                gradient.addColorStop(0, '#0a0a0a');
                gradient.addColorStop(0.3, '#1a1a2e');
                gradient.addColorStop(0.7, '#16213e');
                gradient.addColorStop(1, '#0f3460');
                
                this.ctx.fillStyle = gradient;
                this.ctx.fillRect(0, 0, this.canvas.width / 2, this.canvas.height / 2);
                
                // Add subtle background pattern
                this.ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
                for (let i = 0; i < 50; i++) {{
                    const x = Math.random() * this.canvas.width / 2;
                    const y = Math.random() * this.canvas.height / 2;
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, 1, 0, Math.PI * 2);
                    this.ctx.fill();
                }}
                
                if (this.animationEnabled) {{
                    // Update and draw cells
                    this.cells.forEach(cell => {{
                        cell.pulse += 0.025 * this.animationSpeed;
                        cell.activity = Math.max(0, cell.activity - 0.006);
                        cell.radius = cell.baseRadius + Math.sin(cell.pulse) * (cell.isDragging ? 5 : 3);
                        this.drawCell(cell);
                    }});
                    
                    // Update and draw communications
                    this.communications.forEach(comm => {{
                        comm.progress += comm.speed * this.animationSpeed;
                        if (comm.progress >= 1) {{
                            comm.progress = 0;
                            comm.toCell.activity = Math.min(1, comm.toCell.activity + 0.3);
                        }}
                        this.drawCommunication(comm);
                    }});
                }} else {{
                    // Static mode - just draw cells without animation
                    this.cells.forEach(cell => {{
                        this.drawCell(cell);
                    }});
                    
                    if (this.showCommunications) {{
                        this.communications.forEach(comm => {{
                            this.drawCommunication(comm);
                        }});
                    }}
                }}
                
                // Update stats
                document.getElementById('active-communications').textContent = 
                    `Active Communications: ${{this.communications.length}}`;
                
                requestAnimationFrame(() => this.animate());
            }}
        }}
        
        // Global control functions
        function toggleCommunications() {{
            if (window.visualizer) {{
                window.visualizer.showCommunications = !window.visualizer.showCommunications;
                console.log(`Communications: ${{window.visualizer.showCommunications ? 'ON' : 'OFF'}}`);
            }}
        }}
        
        function resetPositions() {{
            if (window.visualizer) {{
                window.visualizer.cells.forEach(cell => {{
                    cell.x = cell.originalX;
                    cell.y = cell.originalY;
                    cell.activity = 0.5;
                }});
                console.log('Positions reset to original layout');
            }}
        }}
        
        function toggleAnimation() {{
            if (window.visualizer) {{
                window.visualizer.animationEnabled = !window.visualizer.animationEnabled;
                console.log(`Animation: ${{window.visualizer.animationEnabled ? 'ON' : 'OFF'}}`);
            }}
        }}
        
        // Initialize visualizer when page loads
        window.addEventListener('load', () => {{
            console.log('🧬 Initializing Cellular Dependency Visualizer...');
            console.log(`📊 Dependencies: ${{Object.keys(vizData.dependencies).length}}`);
            console.log(`🔗 Relationships: ${{vizData.relationships.length}}`);
            console.log('🎮 Drag cells to move them around!');
            console.log('🖱️  Click dependency buttons to activate communications');
            
            window.visualizer = new CellularDependencyVisualizer();
        }});
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            switch(e.key) {{
                case 'c':
                case 'C':
                    toggleCommunications();
                    break;
                case 'r':
                case 'R':
                    resetPositions();
                    break;
                case 'a':
                case 'A':
                    toggleAnimation();
                    break;
                case ' ':
                    e.preventDefault();
                    // Space bar - activate random cell
                    if (window.visualizer && window.visualizer.cells.length > 0) {{
                        const randomCell = window.visualizer.cells[Math.floor(Math.random() * window.visualizer.cells.length)];
                        window.visualizer.activateCell(randomCell.id);
                    }}
                    break;
            }}
        }});
    </script>
</body>
</html>'''

        return html_content

    def save_results(self, viz_data: Dict, html_content: str):
        """Save all analysis results"""
        print(f"💾 Saving analysis results...")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save visualization data as JSON
        data_file = self.output_dir / "visualization_data.json"
        with open(data_file, 'w') as f:
            json.dump(viz_data, f, indent=2)

        # Save HTML visualization
        html_file = self.output_dir / "cellular_dependencies.html"
        with open(html_file, 'w') as f:
            f.write(html_content)

        # Save analysis report
        self._generate_analysis_report(viz_data)

        print(f"✅ Results saved to: {self.output_dir}")
        print(f"🌐 HTML Visualization: {html_file}")
        print(f"📊 Data File: {data_file}")

        return html_file

    def _generate_analysis_report(self, viz_data: Dict):
        """Generate a comprehensive analysis report"""
        report_content = f"""# Go Cellular Dependency Analysis Report

## Repository Information
- **Repository**: {viz_data['repo_url']}
- **Module**: {viz_data['module_name']}
- **Go Version**: {viz_data['go_version']}
- **Analysis Date**: {viz_data['analysis_timestamp']}

## Dependency Statistics
- **Total Dependencies**: {viz_data['stats']['total_dependencies']}
- **Direct Dependencies**: {viz_data['stats']['direct_dependencies']}
- **Indirect Dependencies**: {viz_data['stats']['indirect_dependencies']}
- **Transitive Dependencies**: {viz_data['stats']['transitive_dependencies']}
- **Total Relationships**: {viz_data['stats']['total_relationships']}

## Dependencies by Category

"""

        for category, deps in viz_data['categories'].items():
            if deps:
                report_content += f"### {category.title()} ({len(deps)} dependencies)\n"
                for dep in deps[:5]:  # Show first 5
                    report_content += f"- **{dep['name']}** `{dep['version']}` ({dep['type']})\n"
                if len(deps) > 5:
                    report_content += f"- ... and {len(deps) - 5} more\n"
                report_content += "\n"

        report_content += f"""
## Visualization Features

The generated cellular visualization includes:

- **🧬 Living Cells**: Each dependency is represented as a biological cell
- **🎨 Color Coding**: Different categories have unique color schemes
- **📏 Size Coding**: Direct dependencies are larger than indirect/transitive
- **🖱️ Draggable Interface**: Click and drag cells to rearrange them
- **📡 Live Communication**: Golden particles show data flow between dependencies
- **🎮 Interactive Controls**: Activate specific dependencies to see their connections
- **⌨️ Keyboard Shortcuts**: 
  - `C` - Toggle communications
  - `R` - Reset positions
  - `A` - Toggle animation
  - `Space` - Activate random cell

## Usage Instructions

1. Open `cellular_dependencies.html` in a web browser
2. Drag cells around to explore the dependency network
3. Click on dependency buttons in the control panel to activate communications
4. Use keyboard shortcuts for quick controls
5. Watch the real-time communication between cellular dependencies

## Technical Details

- **Analysis Method**: Combines `go list`, `go mod graph`, and package import analysis
- **Dependency Types**: 
  - Direct: Listed in go.mod as non-indirect
  - Indirect: Listed in go.mod as indirect
  - Transitive: Discovered through import analysis
- **Categorization**: Intelligent keyword-based categorization
- **Visualization**: HTML5 Canvas with interactive JavaScript

---

*Generated by Go Cellular Dependency Analyzer*
"""

        report_file = self.output_dir / "analysis_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary files")

    def run_analysis(self) -> bool:
        """Run the complete analysis pipeline"""
        try:
            print(f"🚀 Starting analysis of: {self.repo_url}")

            # Step 1: Clone repository
            if not self.clone_repository():
                return False

            # Step 2: Analyze Go module
            if not self.analyze_go_module():
                return False

            # Step 3: Extract dependencies
            if not self.extract_dependencies():
                return False

            # Step 4: Create visualization data
            viz_data = self.create_visualization_data()

            # Step 5: Generate HTML visualization
            html_content = self.generate_html_visualization(viz_data)

            # Step 6: Save results
            html_file = self.save_results(viz_data, html_content)

            print(f"🎉 Analysis completed successfully!")
            print(f"🧬 Found {len(self.dependencies)} dependencies")
            print(f"🔗 Mapped {len(self.relationships)} relationships")
            print(f"🎨 Generated interactive visualization: {html_file}")

            return True

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Analyze Go repositories and create interactive cellular dependency visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python go_cellular_analyzer.py https://github.com/aerospike/aerospike-kubernetes-operator
  python go_cellular_analyzer.py https://github.com/kubernetes/kubernetes --output ./k8s_analysis
  python go_cellular_analyzer.py https://github.com/prometheus/prometheus --output ./prometheus_analysis

Features:
  🧬 Interactive cellular visualization with draggable cells
  📊 Analyzes direct, indirect, and transitive dependencies
  🎨 Color-coded categories (Kubernetes, Database, Web, etc.)
  📡 Live cell-to-cell communication visualization
  🎮 Interactive controls and keyboard shortcuts
  
Keyboard Shortcuts in Visualization:
  C - Toggle communications
  R - Reset cell positions
  A - Toggle animation
  Space - Activate random cell
        """
    )

    parser.add_argument(
        'repo_url',
        nargs='?',
        help='GitHub repository URL to analyze'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output directory for results (default: ./cellular_analysis_<repo_name>)'
    )

    args = parser.parse_args()

    # Interactive mode if no URL provided
    if not args.repo_url:
        print("🧬 Go Cellular Dependency Analyzer")
        print("=" * 50)
        print("Enter a GitHub repository URL to analyze:")
        print("Examples:")
        print("  https://github.com/aerospike/aerospike-kubernetes-operator")
        print("  https://github.com/kubernetes/kubernetes")
        print("  https://github.com/prometheus/prometheus")
        print()

        repo_url = input("Repository URL: ").strip()
        if not repo_url:
            print("❌ No repository URL provided")
            sys.exit(1)
    else:
        repo_url = args.repo_url

    # Validate URL
    if not repo_url.startswith(('http://', 'https://')):
        print("❌ Invalid repository URL. Must start with http:// or https://")
        sys.exit(1)

    # Run analysis
    analyzer = GoCellularAnalyzer(repo_url, args.output)

    success = analyzer.run_analysis()

    if success:
        print(f"")
        print(f"🎉 SUCCESS! Analysis completed successfully!")
        print(f"")
        print(f"📁 Output Directory: {analyzer.output_dir}")
        print(f"🌐 Visualization: {analyzer.output_dir}/cellular_dependencies.html")
        print(f"📊 Data: {analyzer.output_dir}/visualization_data.json")
        print(f"📋 Report: {analyzer.output_dir}/analysis_report.md")
        print(f"")
        print(f"🚀 To view the visualization:")
        print(f"   cd {analyzer.output_dir}")
        print(f"   python3 -m http.server 8000")
        print(f"   Open: http://localhost:8000/cellular_dependencies.html")
        print(f"")
        print(f"🎮 Features in the visualization:")
        print(f"   🖱️  Drag cells to move them around")
        print(f"   🎯 Click dependency buttons to activate communications")
        print(f"   ⌨️  Use keyboard shortcuts (C, R, A, Space)")
        print(f"   📡 Watch live cell-to-cell communication")
        print(f"   🧬 Enjoy the biological cellular experience!")

        sys.exit(0)
    else:
        print(f"❌ Analysis failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

    

