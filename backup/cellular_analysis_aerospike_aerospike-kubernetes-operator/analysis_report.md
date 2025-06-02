# Go Cellular Dependency Analysis Report

## Repository Information
- **Repository**: https://github.com/aerospike/aerospike-kubernetes-operator
- **Module**: github.com/aerospike/aerospike-kubernetes-operator/v4
- **Go Version**: 1.23.0
- **Analysis Date**: 2025-06-02 02:17:00

## Dependency Statistics
- **Total Dependencies**: 281
- **Direct Dependencies**: 20
- **Indirect Dependencies**: 261
- **Transitive Dependencies**: 0
- **Total Relationships**: 1556

## Dependencies by Category

### Kubernetes (25 dependencies)
- **github.com/aerospike/aerospike-client-go/v8** `v8.2.1` (direct)
- **k8s.io/api** `v0.31.0` (direct)
- **k8s.io/apiextensions-apiserver** `v0.31.0` (indirect)
- **k8s.io/apimachinery** `v0.31.0` (direct)
- **k8s.io/apiserver** `v0.31.0` (indirect)
- ... and 20 more

### Cloud (34 dependencies)
- **cloud.google.com/go** `v0.118.3` (indirect)
- **cloud.google.com/go/auth** `v0.15.0` (indirect)
- **cloud.google.com/go/auth/oauth2adapt** `v0.2.8` (indirect)
- **cloud.google.com/go/compute/metadata** `v0.6.0` (indirect)
- **cloud.google.com/go/iam** `v1.4.1` (indirect)
- ... and 29 more

### Database (12 dependencies)
- **github.com/aerospike/aerospike-backup-service/v3** `v3.0.1` (direct)
- **github.com/aerospike/aerospike-management-lib** `v1.7.1-0.20250519063642-57d55e3eddf8` (direct)
- **github.com/aerospike/backup-go** `v0.3.2-0.20250330113002-7fb1b5be7ffc` (indirect)
- **github.com/aerospike/tools-common-go** `v0.2.1-0.20250130070321-acda09110e14` (indirect)
- **go.etcd.io/bbolt** `v1.3.9` (indirect)
- ... and 7 more

### Web (19 dependencies)
- **github.com/emicklei/go-restful/v3** `v3.11.0` (indirect)
- **github.com/felixge/httpsnoop** `v1.0.4` (indirect)
- **github.com/gorilla/mux** `v1.8.1` (indirect)
- **github.com/gorilla/websocket** `v1.5.0` (indirect)
- **github.com/gregjones/httpcache** `v0.0.0-20180305231024-9cad4c3443a7` (indirect)
- ... and 14 more

### Logging (11 dependencies)
- **github.com/chzyer/logex** `v1.1.10` (indirect)
- **github.com/containerd/log** `v0.1.0` (indirect)
- **github.com/go-kit/log** `v0.2.1` (indirect)
- **github.com/go-logfmt/logfmt** `v0.5.1` (indirect)
- **github.com/go-logr/logr** `v1.4.2` (direct)
- ... and 6 more

### Testing (5 dependencies)
- **github.com/kisielk/errcheck** `v1.5.0` (indirect)
- **github.com/onsi/gomega** `v1.36.2` (direct)
- **github.com/stretchr/testify** `v1.10.0` (direct)
- **go.uber.org/mock** `v0.3.0` (indirect)
- **gopkg.in/check.v1** `v1.0.0-20201130134442-10cb98267c6c` (indirect)

### Crypto (7 dependencies)
- **github.com/cespare/xxhash/v2** `v2.3.0` (indirect)
- **github.com/golang-jwt/jwt/v4** `v4.5.0` (indirect)
- **github.com/golang-jwt/jwt/v5** `v5.2.1` (indirect)
- **github.com/googleapis/enterprise-certificate-proxy** `v0.3.6` (indirect)
- **github.com/hashicorp/hcl** `v1.0.0` (indirect)
- ... and 2 more

### Networking (9 dependencies)
- **github.com/GoogleCloudPlatform/opentelemetry-operations-go/exporter/metric** `v0.51.0` (indirect)
- **github.com/GoogleCloudPlatform/opentelemetry-operations-go/internal/resourcemapping** `v0.51.0` (indirect)
- **github.com/envoyproxy/go-control-plane** `v0.13.4` (indirect)
- **github.com/envoyproxy/go-control-plane/envoy** `v1.32.4` (indirect)
- **github.com/envoyproxy/go-control-plane/ratelimit** `v0.1.0` (indirect)
- ... and 4 more

### Data (23 dependencies)
- **github.com/evanphx/json-patch** `v4.12.0+incompatible` (direct)
- **github.com/evanphx/json-patch/v5** `v5.9.0` (indirect)
- **github.com/exponent-io/jsonpath** `v0.0.0-20151013193312-d6023ce2651d` (indirect)
- **github.com/go-openapi/jsonpointer** `v0.19.6` (indirect)
- **github.com/go-openapi/jsonreference** `v0.20.2` (indirect)
- ... and 18 more

### Monitoring (4 dependencies)
- **github.com/prometheus/client_golang** `v1.20.5` (indirect)
- **github.com/prometheus/client_model** `v0.6.1` (indirect)
- **github.com/prometheus/common** `v0.55.0` (indirect)
- **github.com/prometheus/procfs** `v0.15.1` (indirect)

### Storage (2 dependencies)
- **github.com/peterbourgon/diskv** `v2.0.1+incompatible` (indirect)
- **github.com/swaggo/files/v2** `v2.0.0` (indirect)

### Concurrency (1 dependencies)
- **golang.org/x/sync** `v0.12.0` (indirect)

### Cli (3 dependencies)
- **github.com/jessevdk/go-flags** `v1.4.0` (indirect)
- **github.com/spf13/cobra** `v1.9.1` (indirect)
- **github.com/spf13/pflag** `v1.0.6` (indirect)

### Config (3 dependencies)
- **github.com/antihax/optional** `v1.0.0` (indirect)
- **github.com/spf13/viper** `v1.19.0` (indirect)
- **github.com/subosito/gotenv** `v1.6.0` (indirect)

### Utilities (5 dependencies)
- **github.com/NYTimes/gziphandler** `v1.1.1` (indirect)
- **github.com/go-errors/errors** `v1.4.2` (indirect)
- **github.com/pkg/errors** `v0.9.1` (indirect)
- **golang.org/x/time** `v0.11.0` (indirect)
- **golang.org/x/xerrors** `v0.0.0-20220907171357-04be3eba64a2` (indirect)

### External (118 dependencies)
- **cel.dev/expr** `v0.19.2` (indirect)
- **github.com/KyleBanks/depth** `v1.2.1` (indirect)
- **github.com/MakeNowJust/heredoc** `v1.0.0` (indirect)
- **github.com/Microsoft/go-winio** `v0.6.2` (indirect)
- **github.com/alecthomas/kingpin/v2** `v2.4.0` (indirect)
- ... and 113 more


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
