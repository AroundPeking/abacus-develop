# EXX Occupied-Product THC Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 66 服务器的 GaAs TZDP、Gamma-centered $4\times4\times4$ 案例上，判断占据乘积空间是否能在保持 EXX 精度的同时达到至少 $10\times$ 存储压缩和 $5\times$ 因子化核心收缩加速。

**Architecture:** ABACUS 仅增加默认关闭的二进制快照与重放入口；(C,V,D) 保存实际进入 `cal_loop3` 的 active map，(V,D) 另存筛选前 raw map。先用 active (C,V,D) 验证原样重放；再把 raw (D) 只做周期化而不做 filter，建立严格 occupied-factor 参考，避免把块筛选后可能非半正定的 (D_{\rm active}) 错写成 (OO^\dagger)。raw (V) 只用于构造完整 Coulomb 白化 metric，实际 (H,E) 重放和核心计时仍用生产 active (V)。无 MPI 假设的 Python 原型精确构造占据投影后的 $\bar C$、TT Schmidt 谱与代数 occupied-THC。近似张量先重构成只在占据子空间有意义的 $C P_{\rm occ}$，交回原始 `RI::Exx` 重放，以真实 LibRI 路径检验 $H_x,E_x$；性能测试使用 $K=-X[Z\odot S]X^\dagger$ 且禁止恢复稠密 $\bar C$。本计划只覆盖设计稿的阶段 A--C；只有通过 $10\times/5\times$ 门槛后，才另写 LibRI/local-ISDF 接入计划并测试 `cal_exx_elec` 的 $2\times$ 门槛。

**Tech Stack:** ABACUS C++17、LibRI、MPI/OpenMP、CMake/CTest/GoogleTest、Python 3、NumPy、SciPy、`unittest`、Slurm、XeLaTeX。

---

## 0. 范围、公式和硬门槛

本计划实现并验证以下精确变换。对每个自旋和 BvK $\mathbf k$ 点，先对厄米半正定密度矩阵分解

\[
D^\sigma(\mathbf k)=O^\sigma(\mathbf k)O^\sigma(\mathbf k)^\dagger,
\qquad
O=U\sqrt{f}.
\]

令 $O^+$ 为保留占据子空间上的 Moore--Penrose 逆，$P_{\rm occ}=OO^+=UU^\dagger$。把 $C$ 看作从第二个 AO 指标到复合指标 $(\mu i)$ 的线性映射，则

\[
\bar C(\mathbf k)=C(\mathbf k)O(\mathbf k),
\qquad
C_{\rm occ}(\mathbf k)=\bar C(\mathbf k)O(\mathbf k)^+=C(\mathbf k)P_{\rm occ}(\mathbf k),
\]

并且

\[
C_{\rm occ}D C_{\rm occ}^\dagger=CDC^\dagger.
\]

这个等式是“零误差占据投影”硬门槛；它必须先通过小体系的实际 `RI::Exx` 重放，才能用于 GaAs。主候选表示为

\[
\bar C_{\mu iv}\approx\sum_{x=1}^{R}T_{\mu x}^{*}X_{ix}Y_{vx},
\]

直接收缩为

\[
Z=T^\dagger VT,
\qquad
S=Y^T Y^*,
\qquad
K=-X[Z\odot S]X^\dagger.
\]

第一阶段的代数 CP/THC 只回答“这种可分离结构是否存在”，不称为实空间 ISDF。真正的局域插值点、轨道网格值和 LibRI 直接入口属于通过门槛后的第二份计划。

硬门槛固定为：

- 小体系二进制重放的 $H_x,E_x$ 与原运行相对误差均不超过 $10^{-12}$；
- 零误差占据投影经 `RI::Exx` 重放后，
  \(\|H_x^{\rm occ}-H_x\|_F/\|H_x\|_F\le10^{-10}\)，
  \(|E_x^{\rm occ}-E_x|\le10^{-10}\) Ry/atom；
- GaAs 近似表示满足
  \(\|\widehat H_x-H_x\|_F/\|H_x\|_F\le10^{-4}\)，
  \(|\widehat E_x-E_x|\le1\) meV/atom；
- 全部因子、索引、全部 $\mathbf k/\mathbf q$ 扇区的 $Z$ 与最大工作区合计至少压缩 $10\times$；
- 同节点、同线程数、不解压的因子化核心至少加速 $5\times$；
- 因子生成成本在 5 次 EXX 更新内摊销；否则结论注明“压缩存在但计算不经济”。

带边 $10$ meV 与 `cal_exx_elec` $2\times$ 属于第二阶段生产接入验收，不用第一阶段的离线微基准代替。

## 1. 文件与职责

本计划创建或修改以下文件；不得把这些职责重新塞进 `Exx_LRI.hpp`：

- `source/source_lcao/module_ri/exx_compression_io.h`：版本化二进制 `RI::Tensor` map 读写，显式实数/复数编码。
- `source/source_lcao/module_ri/exx_compression_dump.h`：环境变量开关、路径命名、active (C,V,D) 与 raw (V,D) 快照调度。
- `source/source_lcao/module_ri/Exx_LRI.hpp`：只在已有 (C,V,D,H,E) 生命周期位置调用 dump；默认路径不改变。
- `source/source_lcao/module_ri/test/exx_compression_io_test.cpp`：二进制格式、损坏输入、复数往返测试。
- `source/source_lcao/module_ri/test/exx_compression_replay.cpp`：单 MPI rank 读取快照，调用原始 `RI::Exx`，写出重放 (H,E)。
- `source/source_lcao/module_ri/test/CMakeLists.txt`：注册上述测试和重放工具。
- `tools/exx_thc/pyproject.toml`：离线包元数据，仅声明服务器已有的 NumPy/SciPy。
- `tools/exx_thc/src/exx_thc/io.py`：解析/写回 `EXXCMP1` 文件及清单。
- `tools/exx_thc/src/exx_thc/bvk.py`：原子块组装、离散 Fourier 变换、Hermiticity 与周期往返检查。
- `tools/exx_thc/src/exx_thc/occupied.py`：(D=OO^\dagger)、(O^+)、(\bar C)、(C P_{\rm occ}) 和占据截断。
- `tools/exx_thc/src/exx_thc/tt.py`：三阶 TT-SVD、Schmidt 谱、discarded weight 和误差界。
- `tools/exx_thc/src/exx_thc/thc.py`：确定性的复数 CP-ALS、THC 因子约定和稠密重构，仅供精度验证。
- `tools/exx_thc/src/exx_thc/contract.py`：稠密占据收缩与不解压 THC 收缩。
- `tools/exx_thc/src/exx_thc/metrics.py`：误差、全量字节数、理论 FLOP、计时统计和门槛判定。
- `tools/exx_thc/src/exx_thc/cli.py`：`inspect`、`compare`、`project`、`scan`、`benchmark` 五个可复现命令。
- `tools/exx_thc/tests/`：与上述模块一一对应的 `unittest`。
- `tools/exx_thc/cases/gaas_k444/`：固定 GaAs 输入、Slurm 脚本和 SHA256 清单；不把 PP/轨道大文件提交进 Git。
- `/Users/ghj/同步空间/AITP_project/exx_occupied_thc_feasibility/`：最终 TeX/PDF、输入、数据摘要、图、脚本副本和来源记录。

## Task 1: 冻结版本、构建环境与 GaAs 输入

**Files:**
- Create: `tools/exx_thc/cases/gaas_k444/STRU`
- Create: `tools/exx_thc/cases/gaas_k444/KPT`
- Create: `tools/exx_thc/cases/gaas_k444/INPUT_pbe`
- Create: `tools/exx_thc/cases/gaas_k444/INPUT_exx_snapshot`
- Create: `tools/exx_thc/cases/gaas_k444/run_snapshot.slurm`
- Create: `tools/exx_thc/cases/gaas_k444/README.md`
- Create: `tools/exx_thc/cases/gaas_k444/checksums.sha256`

- [ ] **Step 1: 记录本地、66 源码和依赖版本**

在本地执行：

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse master_ghj
```

在 66 上使用旧 Git 兼容写法：

```bash
ssh 159.226.208.66 'bash -lc '\''
source ~/.bashrc >/dev/null 2>&1 || true
cd /home/ghj/abacus/260810/mps-exx-k444/source
git status --short --branch
git rev-parse HEAD
git --version
python3 -c "import numpy,scipy,threadpoolctl; print(numpy.__version__,scipy.__version__,threadpoolctl.__version__)"
'\'''
```

Expected: 本地和远端分支均为 `codex/mps-exx-k444`、工作区干净；远端 Python 显示 NumPy `1.23.3`、SciPy `1.9.1`、threadpoolctl `3.1.0`。若 SHA 不同，先按 Task 12 的同步步骤快进远端，不运行计算。

- [ ] **Step 2: 写固定晶体与 k 点文件**

`STRU` 的完整内容为：

```text
ATOMIC_SPECIES
Ga 69.723000 Ga.upf
As 74.921595 As.upf

NUMERICAL_ORBITAL
Ga_gga_8au_100Ry_3s3p3d2f.orb
As_gga_8au_100Ry_3s3p3d2f.orb

LATTICE_CONSTANT
10.683094214040635

LATTICE_VECTORS
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0

ATOMIC_POSITIONS
Direct

Ga
0.0
1
0.0 0.0 0.0 0 0 0

As
0.0
1
0.25 0.25 0.25 0 0 0
```

`KPT` 的完整内容为：

```text
K_POINTS
0
Gamma
4 4 4 0 0 0
```

Expected: 两原子原胞、82 个 NAO、Gamma-centered 64 个 k 点。

- [ ] **Step 3: 写 PBE 和一次 EXX 快照输入**

`INPUT_pbe`：

```text
INPUT_PARAMETERS
suffix                    GAAS_TZDP_K444_PBE
calculation               scf
ntype                     2
symmetry                  0
gamma_only                0
nspin                     1
basis_type                lcao
ks_solver                 genelpa
ecutwfc                   100
scf_thr                   1e-9
scf_nmax                  100
smearing_method           fixed
mixing_type               broyden
mixing_beta               0.4
dft_functional            pbe
out_chg                   1
out_wfc_lcao              1
pseudo_dir                ./
orbital_dir               ./
```

`INPUT_exx_snapshot`：

```text
INPUT_PARAMETERS
suffix                    GAAS_TZDP_K444_EXX_SNAPSHOT
calculation               scf
ntype                     2
symmetry                  0
gamma_only                0
nspin                     1
basis_type                lcao
ks_solver                 genelpa
ecutwfc                   100
scf_thr                   1e-9
scf_nmax                  100
smearing_method           fixed
mixing_type               broyden
mixing_beta               0.4
dft_functional            pbe0
exx_real_number           1
exx_pca_threshold         1e-4
exx_c_threshold           1e-4
exx_v_threshold           0.1
exx_dm_threshold          1e-4
exx_ccp_rmesh_times       2
exx_separate_loop         1
exx_hybrid_step           1
out_chg                   1
out_wfc_lcao              1
pseudo_dir                ./
orbital_dir               ./
```

Expected: EXX 输入先完成半局域内循环，只保存第一次固定密度矩阵的 EXX 快照；不把它解释为收敛 PBE0 总能。这里固定生产安全默认的 (C/V/D) 筛选阈值，occupied-THC 阈值另行扫描；不得通过收紧原始 LibRI 筛选、人为放大稠密基线来制造加速比。

- [ ] **Step 4: 写服务器资源脚本**

`run_snapshot.slurm`：

```bash
#!/bin/bash
#SBATCH --job-name=gaas_k444_exxcmp
#SBATCH --partition=640
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=175000
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

set -euo pipefail
abacus_exe=/home/ghj/abacus/260810/mps-exx-k444/build/abacus_3p
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export ABACUS_EXX_COMPRESSION_DUMP="$PWD/exxcmp"

date
hostname
pwd
printf 'OMP_NUM_THREADS=%s\n' "$OMP_NUM_THREADS"
"$abacus_exe"
date
```

提交前用 `sinfo -p 640 -o '%P %a %D %c %m %N'` 重查资源；如果节点内存不再是约 180000 MB，调整 `--mem` 为不超过实时值的明确数字。2026-08-11 实测 `ghj` 账户不允许使用 740，因此不能根据 740 的空闲节点状态提交。

- [ ] **Step 5: 固定四个大文件的来源与哈希**

复制而不是链接下列文件到远端 run 目录，并写入 `checksums.sha256`：

```text
e890f0d10ab5554c3c6440f8a617507740e625ca5a28f266d6fec4260a9d0723  Ga.upf
39d557b0496b28f758c2e3e5c0bafa35c8095b7fd72ec30cb3f912c3f0c0f68f  As.upf
da292444d04f489d2f18c3f8b9a1afefbb51ef26891359cdde45314284d58bd4  Ga_gga_8au_100Ry_3s3p3d2f.orb
543016e20180a8cbdce51cfca308e750e1edbaf4720e0defda719360ae580d12  As_gga_8au_100Ry_3s3p3d2f.orb
```

本地来源固定为 `/Users/ghj/同步空间/AITP_project/sternheimer_abacus/ABACUS-orbitals/Dojo-NC-SR/`。远端执行 `sha256sum -c checksums.sha256`，Expected: 四行 `OK`。

- [ ] **Step 6: 提交输入资产**

```bash
git add tools/exx_thc/cases/gaas_k444
git commit -m "test: define GaAs k444 EXX compression case"
```

提交时必须使用 Codex author、AroundPeking committer，并用 `git show -s --format=fuller HEAD` 验证。

## Task 2: 实现版本化二进制 tensor-map I/O

**Files:**
- Create: `source/source_lcao/module_ri/exx_compression_io.h`
- Create: `source/source_lcao/module_ri/test/exx_compression_io_test.cpp`
- Modify: `source/source_lcao/module_ri/test/CMakeLists.txt`

- [ ] **Step 1: 先写实数、复数和损坏 magic 的失败测试**

测试必须构造键 `(ia1, ia2, R)` 和非方形 shape，检查所有值逐元素相等：

```cpp
TEST(ExxCompressionIO, ComplexMapRoundTrip)
{
    ExxCompressionIO::TensorMap<std::complex<double>> input;
    RI::Tensor<std::complex<double>> t({2, 2, 3});
    for (std::size_t i = 0; i < t.get_shape_all(); ++i)
        t.ptr()[i] = {static_cast<double>(i), -0.25 * static_cast<double>(i)};
    input[1][{0, {2, -1, 0}}] = t;
    ExxCompressionIO::write_map("complex.exxcmp", input, 0, 1);
    const auto output = ExxCompressionIO::read_map<std::complex<double>>("complex.exxcmp");
    ASSERT_EQ(output.at(1).at({0, {2, -1, 0}}).shape, t.shape);
    for (std::size_t i = 0; i < t.get_shape_all(); ++i)
        EXPECT_EQ(output.at(1).at({0, {2, -1, 0}}).ptr()[i], t.ptr()[i]);
}
```

另写 `BadMagicThrows`，把首字节改为 `X` 后要求 `std::runtime_error`。

- [ ] **Step 2: 运行测试确认它因头文件不存在而失败**

```bash
cmake --build build --target MODULE_RI_exx_compression_io_test -j8
```

Expected: FAIL，错误包含 `exx_compression_io.h: No such file or directory`。

- [ ] **Step 3: 实现固定 schema**

文件严格使用 little-endian、显式标量编码：

```cpp
namespace ExxCompressionIO
{
using TC = std::array<int, 3>;
using TAC = std::pair<int, TC>;
template<class T>
using TensorMap = std::map<int, std::map<TAC, RI::Tensor<T>>>;

struct Header
{
    char magic[8];          // "EXXCMP1\0"
    std::uint32_t version;  // 1
    std::uint8_t scalar;    // 1=real64, 2=complex128
    std::uint8_t reserved[3];
    std::uint32_t rank;
    std::uint32_t nranks;
    std::uint64_t records;
};

template<class T>
void write_map(const std::string& path, const TensorMap<T>& map, int rank, int nranks);

template<class T>
TensorMap<T> read_map(const std::string& path);
}
```

Header 和 record 都逐字段写入，不得用 `sizeof(Header)` 直接写 struct，以免把 padding 带入文件。每条 record 依次写 `int32 ia1, ia2, R[3]`、`uint32 ndim`、`uint64 shape[ndim]`、`uint64 value_count`。`double` 写一个 IEEE754 `float64`；`complex<double>` 对每个元素显式写 `real, imag` 两个 `float64`，不得直接 dump `std::complex` 内存布局。读取时验证 magic、version、scalar、`ndim in [1,4]`、shape 乘积与 `value_count` 一致、文件无截断。

- [ ] **Step 4: 注册并运行测试**

在 `source/source_lcao/module_ri/test/CMakeLists.txt` 增加：

```cmake
AddTest(
  TARGET MODULE_RI_exx_compression_io_test
  LIBS base ${math_libs} device parameter
  SOURCES exx_compression_io_test.cpp
)
```

运行：

```bash
cmake --build build --target MODULE_RI_exx_compression_io_test -j8
ctest --test-dir build -R MODULE_RI_exx_compression_io_test --output-on-failure
```

Expected: `100% tests passed`。

- [ ] **Step 5: 提交**

```bash
git add source/source_lcao/module_ri/exx_compression_io.h \
        source/source_lcao/module_ri/test/exx_compression_io_test.cpp \
        source/source_lcao/module_ri/test/CMakeLists.txt
git commit -m "test: add EXX tensor-map snapshot format"
```

## Task 3: 接入默认关闭的 (C,V,D,H,E) 快照

**Files:**
- Create: `source/source_lcao/module_ri/exx_compression_dump.h`
- Modify: `source/source_lcao/module_ri/Exx_LRI.hpp`
- Modify: `source/source_lcao/module_ri/test/exx_compression_io_test.cpp`

- [ ] **Step 1: 写环境变量和命名测试**

测试要求未设置环境变量时 `enabled()==false`；设置空串或 `0` 仍为 false；设置绝对目录时文件名含对象、spin、channel、rank：

```cpp
EXPECT_FALSE(ExxCompressionDump::enabled(nullptr));
EXPECT_FALSE(ExxCompressionDump::enabled(""));
EXPECT_FALSE(ExxCompressionDump::enabled("0"));
EXPECT_EQ(
  ExxCompressionDump::map_path("/tmp/snap", "D", 1, "short", 3),
  "/tmp/snap/D.spin1.short.rank000003.exxcmp");
```

- [ ] **Step 2: 实现 dump API**

```cpp
namespace ExxCompressionDump
{
bool enabled(const char* value);
std::string root(); // getenv("ABACUS_EXX_COMPRESSION_DUMP")
std::string map_path(const std::string&, const std::string&, int,
                     const std::string&, int);

template<class T>
void write_if_enabled(const std::string& object,
                      const ExxCompressionIO::TensorMap<T>& map,
                      int spin, const std::string& channel, MPI_Comm comm);

template<class T>
void write_scalar_if_enabled(const std::string& object, const T& value,
                             int spin, const std::string& channel, MPI_Comm comm);
}
```

只有 rank 0 创建目录和写 `manifest.txt`；所有 rank 写自己的 map 文件。`manifest.txt` 至少包含 ABACUS commit、scalar、MPI size、period、latvec、原子位置、(C/V/D) 阈值、spin、channel 和时间。目录已存在时只允许追加新的、名称不冲突的 channel；不得覆盖已有 `.exxcmp`。

标量能量文件用 `std::setprecision(17)` 写实部和虚部两列；Python 比较时要求虚部绝对值 `<1e-13`，否则把快照判为异常。

- [ ] **Step 3: 在真实生命周期点调用，不改变默认路径**

在 `Exx_LRI::cal_exx_ions` 中，每次 `set_Cs` 返回后从 `data_pool` 写 active `C`；每次 `set_Vs` 前写 `V.raw`，返回后从 `data_pool` 写 `V.active`。在 `cal_exx_elec` 的 `run_exx_channel` lambda 中，先写筛选前 `D.raw`，`set_Ds` 返回后再写 `D.active`。`cal_Hs` 返回后、`Hs` 被 move 或对称性恢复之前写 `H.lri` 与 `E.lri`。外层 `post_process_Hexx/Eexx` 完成后另写 `H.final/E.final`，用于确认 ABACUS 的 (-2) Hamiltonian 因子和自旋能量因子。核心调用形状固定为：

```cpp
ExxCompressionDump::write_if_enabled(
    "C.active", exx_channel.lri.data_pool.at("Cs_"+suffix).Ds_ab,
    -1, channel, this->mpi_comm);
ExxCompressionDump::write_if_enabled(
    "V.raw", V_in, -1, channel, this->mpi_comm);
ExxCompressionDump::write_if_enabled(
    "V.active", exx_channel.lri.data_pool.at("Vs_"+suffix).Ds_ab,
    -1, channel, this->mpi_comm);
ExxCompressionDump::write_if_enabled(
    "D.raw", D_in, is, channel, this->mpi_comm);
ExxCompressionDump::write_if_enabled(
    "D.active", exx_channel.lri.data_pool.at("Ds_"+suffix).Ds_ab,
    is, channel, this->mpi_comm);
ExxCompressionDump::write_if_enabled(
    "H.lri", exx_channel.Hs, is, channel, this->mpi_comm);
ExxCompressionDump::write_scalar_if_enabled(
    "E.lri", exx_channel.energy, is, channel, this->mpi_comm);
```

其中 `V_in` 是传给当前 `set_Vs` 的、尚未 move 的 map；`exx_channel` 表示当前 full/short/long 的 `RI::Exx` 对象，`suffix` 必须与对应 `set_*` 调用完全一致。实现时以当前源码实参名替换，不改变计算顺序。`H.final/E.final` 在 `this->Hexxs` 和 `this->Eexx` 完成现有后处理后写出。所有调用前先在函数内缓存一次 `enabled`，关闭时不遍历 map。压缩字节数只与 active map 比较；raw (V,D) 的字节不计入候选压缩收益，也不作为稠密基线。

- [ ] **Step 4: 验证关闭开关时集成测试完全不变**

```bash
ctest --test-dir build -R 'MODULE_RI_exx_compression|08_EXX' --output-on-failure
```

Expected: 新单元测试通过；`tests/08_EXX/11_KP_PBE0/result.ref` 对应的能量与迭代结果不变，且未生成 `exxcmp` 目录。

- [ ] **Step 5: 验证开启开关的小 Si 快照**

以 `tests/08_EXX/11_KP_PBE0` 的副本运行 1 MPI rank，设置：

```bash
export ABACUS_EXX_COMPRESSION_DUMP="$PWD/exxcmp"
```

Expected: `manifest.txt`、`C.active.*.exxcmp`、`V.raw.*.exxcmp`、`V.active.*.exxcmp`、`D.raw.spin0.*.exxcmp`、`D.active.spin0.*.exxcmp`、`H.lri.spin0.*.exxcmp`、`E.lri.spin0.*`、`H.final.spin0.*.exxcmp`、`E.final.*` 均存在；manifest 分开记录 raw/active (V,D) 的块数和字节数；关闭开关重跑后不生成这些文件。

- [ ] **Step 6: 提交**

```bash
git add source/source_lcao/module_ri/exx_compression_dump.h \
        source/source_lcao/module_ri/Exx_LRI.hpp \
        source/source_lcao/module_ri/test/exx_compression_io_test.cpp
git commit -m "feat: add opt-in EXX compression snapshots"
```

## Task 4: Python 解析、BvK 变换和占据投影

**Files:**
- Create: `tools/exx_thc/pyproject.toml`
- Create: `tools/exx_thc/src/exx_thc/__init__.py`
- Create: `tools/exx_thc/src/exx_thc/io.py`
- Create: `tools/exx_thc/src/exx_thc/bvk.py`
- Create: `tools/exx_thc/src/exx_thc/occupied.py`
- Create: `tools/exx_thc/tests/test_io.py`
- Create: `tools/exx_thc/tests/test_bvk.py`
- Create: `tools/exx_thc/tests/test_occupied.py`

- [ ] **Step 1: 写失败测试，覆盖 C++ 文件、Fourier 往返和复数 PSD**

核心测试为：

```python
def test_occupied_projection_preserves_cdc():
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(6, 3)) + 1j*rng.normal(size=(6, 3)))
    occ = np.array([1.0, 0.7, 0.2])
    d = (q * occ) @ q.conj().T
    c = rng.normal(size=(5, 6)) + 1j*rng.normal(size=(5, 6))
    factor = occupied_factor(d, eigenvalue_tol=0.0)
    cbar = c @ factor.O
    c_occ = cbar @ factor.O_pinv
    np.testing.assert_allclose(c_occ @ d @ c_occ.conj().T,
                               c @ d @ c.conj().T, rtol=1e-12, atol=1e-12)
```

`test_bvk.py` 用 period `(2,2,1)` 和非零负晶格矢量，要求 `from_k(to_k(blocks))` 在 `1e-13` 内往返，并逐个 (D(k)) 检查 Hermiticity。

- [ ] **Step 2: 运行并确认失败**

```bash
PYTHONPATH=tools/exx_thc/src python3 -m unittest discover -s tools/exx_thc/tests -v
```

Expected: FAIL，模块 `exx_thc.io`、`bvk`、`occupied` 尚不存在。

- [ ] **Step 3: 实现明确的数据类型和 Fourier 约定**

`pyproject.toml` 的完整内容为：

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "exx-thc"
version = "0.1.0"
requires-python = ">=3.8"
dependencies = ["numpy==1.23.3", "scipy==1.9.1", "threadpoolctl==3.1.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

`io.py` 返回：

```python
@dataclass(frozen=True)
class BlockKey:
    ia1: int
    ia2: int
    R: tuple[int, int, int]

TensorMap = dict[BlockKey, np.ndarray]
```

`bvk.py` 固定采用

```python
phase = np.exp(-2j*np.pi*np.dot(k_index/np.asarray(period), R))
```

正变换，逆变换使用 (1/N_k) 和相反相位。对同一原子对缺失的 (R) 块按零处理；输出一律 `complex128`。`assemble_matrix` 从每个原子块 shape 推导 AO/ABF offset，发现同一原子维数冲突立即抛出 `ValueError`。

`occupied.py` 的公开接口固定为：

```python
@dataclass(frozen=True)
class OccupiedFactor:
    O: np.ndarray
    O_pinv: np.ndarray
    eigenvalues: np.ndarray
    discarded_trace: float

def occupied_factor(d: np.ndarray, eigenvalue_tol: float) -> OccupiedFactor:
    dh = 0.5 * (d + d.conj().T)
    w, u = scipy.linalg.eigh(dh, driver="evr")
    scale = max(float(np.max(np.abs(w))), 1.0)
    if np.min(w) < -1e-10 * scale:
        raise ValueError("density matrix is not positive semidefinite")
    w = np.clip(w, 0.0, None)
    keep = w > eigenvalue_tol * max(float(w.max()), 1.0)
    o = u[:, keep] * np.sqrt(w[keep])
    opinv = (u[:, keep] / np.sqrt(w[keep])).conj().T
    return OccupiedFactor(o, opinv, w, float(w[~keep].sum()))
```

零占据矩阵返回 shape `(n,0)` 和 `(0,n)`，不除以零。

- [ ] **Step 4: 运行全部 Python 单元测试**

```bash
PYTHONPATH=tools/exx_thc/src python3 -m unittest discover -s tools/exx_thc/tests -v
```

Expected: 所有测试 `OK`，不安装 `tensorly` 或修改 base conda。

- [ ] **Step 5: 提交**

```bash
git add tools/exx_thc/pyproject.toml tools/exx_thc/src tools/exx_thc/tests
git commit -m "feat: add occupied-space EXX analysis core"
```

## Task 5: 用原始 `RI::Exx` 做快照重放和零误差物理门槛

**Files:**
- Create: `source/source_lcao/module_ri/test/exx_compression_replay.cpp`
- Modify: `source/source_lcao/module_ri/test/CMakeLists.txt`
- Create: `tools/exx_thc/src/exx_thc/cli.py`
- Create: `tools/exx_thc/tests/test_cli.py`
- Create: `tools/exx_thc/docs/libri_map_semantics.md`

- [ ] **Step 1: 写 replay 的合成失败测试**

构造 period `(2,1,1)`、1 原子、复数 Hermitian (D(R))、Hermitian (V(R)) 和随机 (C(R))，先直接调用 `RI::Exx` 得到 (H,E)，写快照后再由 replay 读取并计算。active 测试必须令用于收缩的 `D.active` 与用于 `post_2D` 能量的原始 `D.raw` 不同，避免 replay 与测试共同重复同一个错误。要求 map key、shape、(H) 元素和 (E) 在 `1e-13` 内相同。

- [ ] **Step 2: 实现 replay 工具**

主流程固定为 active (C,V) 原样装载。`cfg.D_state=="active"` 时，`D_path` 的 active (D) 原样装载、只负责 `cal_Hs` 收缩，另要求 `D_post_path` 指向生产 `set_Ds` 收到的 raw (D)，只负责 `post_2D` 能量；`cfg.D_state=="raw"` 时，`D_path` 同时作为原始 `post_2D` 密度，并只为 `cal_Hs` 设置 `flag_period=true`，仍保持 `flag_comm=false, flag_filter=false`。这是因为固定 LibRI 版本的 `Exx::set_Ds` 用 periodized/filtered data pool 计算 (H)，但用原始传入 (D) 计算 (E)：

```cpp
MPI_Init(&argc, &argv);
const ReplayConfig cfg = read_replay_config(argv[1]);
RI::Exx<int, int, 3, std::complex<double>> exx;
exx.set_parallel(MPI_COMM_WORLD, cfg.atoms_pos, cfg.latvec, cfg.period);
exx.set_symmetry(false, {});
const std::map<std::string, double> active_flags = {
    {"flag_period", false}, {"flag_comm", false}, {"flag_filter", false}};
const std::map<std::string, double> raw_d_flags = {
    {"flag_period", true}, {"flag_comm", false}, {"flag_filter", false}};
const auto Cs = read_map<std::complex<double>>(cfg.C_path);
const auto Vs = read_map<std::complex<double>>(cfg.V_path);
const auto Ds = read_map<std::complex<double>>(cfg.D_path);
exx.lri.set_tensors_map2(Cs, {RI::Label::ab::a, RI::Label::ab::b},
                         active_flags, "Cs_");
exx.lri.set_tensors_map2(Vs, {RI::Label::ab::a0b0},
                         active_flags, "Vs_");
exx.lri.set_tensors_map2(
    Ds, {RI::Label::ab::a1b1, RI::Label::ab::a1b2,
         RI::Label::ab::a2b1, RI::Label::ab::a2b2},
    cfg.D_state == "raw" ? raw_d_flags : active_flags, "Ds_");
exx.flag_finish.Cs = exx.flag_finish.Vs = exx.flag_finish.Ds = true;
const auto& Ds_loaded = exx.lri.data_pool.at("Ds_").Ds_ab;
const auto Ds_post = cfg.D_state == "raw"
    ? Ds
    : read_map<std::complex<double>>(cfg.D_post_path);
exx.post_2D.saves["Ds_"] = exx.post_2D.set_tensors_map2(Ds_post);
exx.cal_Hs();
write_map(cfg.H_out, exx.Hs, 0, 1);
write_scalar(cfg.E_out, exx.energy);
MPI_Finalize();
```

只支持 `nranks=1`；输入快照的 `nranks!=1` 时明确退出，避免把不完整分布式 map 当成全量数据。active 模式强制要求 `D_post_path`，raw 模式强制禁止它。raw-(D) 重放产生的 periodized `Ds_loaded` 另写为 `D.full.exxcmp`，后续 Python 只对这个严格参考执行 (D=OO^\dagger)。所有输出不得覆盖已有文件，先写同目录临时文件，最后独占发布 (E) 作为完成标志；消费者只有在 (H,E) 以及 raw 模式的 `D.full` 全部存在时才接受结果。

- [ ] **Step 3: 注册 replay target 并通过合成测试**

在 `ENABLE_MPI` 块内增加可执行文件，链接 `MPI::MPI_CXX`、LibRI 和 module RI 已用库。运行：

```cmake
AddTest(
  TARGET MODULE_RI_exx_compression_replay
  LIBS ri base parameter ${math_libs} device MPI::MPI_CXX
  SOURCES exx_compression_replay.cpp
)
```

```bash
cmake --build build --target MODULE_RI_exx_compression_replay -j8
ctest --test-dir build -R MODULE_RI_exx_compression_replay --output-on-failure
```

Expected: `100% tests passed`。

- [ ] **Step 4: 写 LibRI 标签和数据语义审计**

`tools/exx_thc/docs/libri_map_semantics.md` 必须记录并链接到当前源码行：

```text
C: set_tensors_map2 labels a,b; raw block shape [naux(ia1), nao(ia1), nao(ia2)]
V: label a0b0; raw block shape [naux(ia1), naux(ia2)]
D: labels a1b1,a1b2,a2b1,a2b2; raw block shape [nao(ia1), nao(ia2)]
H: RI::Exx::cal_Hs result map; energy from Exx_Post_2D::cal_energy(D_post,H), where D_post is the original pre-filter D passed to set_Ds
period: RI::Exx::lri.period; R is interpreted modulo this period after set_tensors_map2
```

同时记录 `RI::Exx::cal_Hs` 是否存在 (D=OO^\dagger) 或占据轨道入口。当前预期结论是：接口只接收 (D) map，`cal_loop3` 中未显式暴露 occupied-factor 路径。

- [ ] **Step 5: 在小 Si 实例做原始快照重放门槛**

对 `tests/08_EXX/11_KP_PBE0` active 快照运行 replay，再用 `cli compare` 比较原始与重放：

```bash
PYTHONPATH=tools/exx_thc/src python3 -m exx_thc.cli compare \
  --reference exxcmp/H.lri.spin0.full.rank000000.exxcmp \
  --candidate replay/H.lri.spin0.full.rank000000.exxcmp \
  --energy-reference exxcmp/E.lri.spin0.full.txt \
  --energy-candidate replay/E.lri.spin0.full.txt
```

active replay 配置使用 `D_path=D.active...` 和 `D_post_path=D.raw...`。Expected: active (C,V,D) 的 `H_rel_fro <= 1e-12`、`E_abs_Ry_atom <= 1e-12`，命令退出 0。这一步只证明快照/重放忠实，不对 active (D) 做 PSD 假设。

- [ ] **Step 6: 做零误差 occupied projection 重放**

先用 raw-(D) replay 生成 `D.full.exxcmp` 及对应 (H_{\rm full},E_{\rm full})。`cli project` 对 `D.full` 的所有 BvK (k) 扇区执行 (D(k)=OO^\dagger)、(C_{\rm occ}(k)=C(k)P_{\rm occ}(k))，逆变换回 `C_occ.active.exxcmp`。首先要求 (C,D) 各自的 Fourier 往返误差 `<1e-13`、每个 (D(k)) Hermiticity `<1e-12`、负本征值不低于 `-1e-10*lambda_max`；再把 `C_occ` 与同一个 `D.full` 作为 active map 交给 replay，同时仍以原始 `D.raw` 作为 `D_post_path` 计算能量。

Expected: 小 Si 相对于 raw-(D) full reference 的 `H_rel_fro <= 1e-10`、`E_abs_Ry_atom <= 1e-10`。另报告生产 active-(D) 与 full-(D) 的差，但不把这部分既有 DM screening 误差归因于 occupied projection。若零误差投影失败，停止 GaAs 和 THC 工作，保留快照，并把问题定位为 Fourier/LibRI map 语义不一致；不得调宽门槛掩盖。

- [ ] **Step 7: 提交**

```bash
git add source/source_lcao/module_ri/test tools/exx_thc/src/exx_thc/cli.py \
        tools/exx_thc/tests/test_cli.py tools/exx_thc/docs/libri_map_semantics.md
git commit -m "test: replay EXX snapshots through LibRI"
```

## Task 6: 实现 Coulomb 白化的 TT/Schmidt 上界

**Files:**
- Create: `tools/exx_thc/src/exx_thc/tt.py`
- Create: `tools/exx_thc/src/exx_thc/metrics.py`
- Create: `tools/exx_thc/tests/test_tt.py`
- Create: `tools/exx_thc/tests/test_metrics.py`

- [ ] **Step 1: 写已知 TT 秩和误差界测试**

人工生成 ranks `(2,3)` 的复数三阶 TT，要求零阈值恢复；再截断到 `(1,1)`，要求实际误差不超过两次 discarded singular-value square 的和：

```python
result = tt_svd_3(tensor, relative_tol=1e-12)
assert result.ranks == (2, 3)
np.testing.assert_allclose(result.reconstruct(), tensor, rtol=1e-11, atol=1e-11)
truncated = tt_svd_3(tensor, relative_tol=0.5)
assert np.linalg.norm(tensor-truncated.reconstruct())**2 <= truncated.error_bound*(1+1e-12)
```

- [ ] **Step 2: 实现三阶 TT-SVD 和 discarded weight**

接口固定为：

```python
@dataclass(frozen=True)
class TT3:
    g1: np.ndarray
    g2: np.ndarray
    g3: np.ndarray
    spectra: tuple[np.ndarray, np.ndarray]
    discarded_weights: tuple[float, float]
    error_bound: float
    ranks: tuple[int, int]

    def reconstruct(self) -> np.ndarray:
        return np.einsum("ia,ajb,bk->ijk", self.g1, self.g2, self.g3)

def tt_svd_3(a: np.ndarray, relative_tol: float) -> TT3:
    if a.ndim != 3:
        raise ValueError("tt_svd_3 expects a rank-3 tensor")
    norm2 = float(np.vdot(a, a).real)
    n1, n2, n3 = a.shape
    if norm2 == 0.0:
        return TT3(np.zeros((n1, 0), dtype=a.dtype),
                   np.zeros((0, n2, 0), dtype=a.dtype),
                   np.zeros((0, n3), dtype=a.dtype),
                   (np.zeros(0), np.zeros(0)), (0.0, 0.0), 0.0, (0, 0))

    def choose_rank(s, budget):
        for rank in range(1, s.size + 1):
            if float(np.dot(s[rank:], s[rank:])) <= budget:
                return rank
        return s.size

    budget = relative_tol**2 * norm2 / 2.0
    u1, s1, vh1 = scipy.linalg.svd(a.reshape(n1, n2*n3),
                                    full_matrices=False, lapack_driver="gesdd")
    r1 = choose_rank(s1, budget)
    g1 = u1[:, :r1]
    remainder = (s1[:r1, None] * vh1[:r1]).reshape(r1*n2, n3)
    u2, s2, vh2 = scipy.linalg.svd(remainder, full_matrices=False,
                                    lapack_driver="gesdd")
    r2 = choose_rank(s2, budget)
    g2 = u2[:, :r2].reshape(r1, n2, r2)
    g3 = s2[:r2, None] * vh2[:r2]
    tail1 = float(np.dot(s1[r1:], s1[r1:]))
    tail2 = float(np.dot(s2[r2:], s2[r2:]))
    return TT3(g1, g2, g3, (s1, s2),
               (tail1/norm2, tail2/norm2), tail1+tail2, (r1, r2))
```

每次 SVD 选择最小秩 (r)，使尾部平方和不超过 `relative_tol**2 * ||a||_F**2 / 2`；保存完整奇异值谱和归一化 discarded weight。零张量返回 ranks `(0,0)`、零误差。

- [ ] **Step 3: 实现物理 Coulomb 白化**

先将 `V.raw` 只做 periodization 并组装每个完整 (q) 扇区；`metrics.whiten(v, cbar)` 再厄米化 `v`，用 `scipy.linalg.eigh`。若本征值低于 `-1e-10*lambda_max` 则失败，处于该范围内的数值负值截成 0，`lambda <= 1e-12*lambda_max` 作为零模删除。构造 (L=U\sqrt{\Lambda})，返回 `einsum('pm,miv->piv', L.conj().T, cbar)` 及删除零模数。不得对单独的非对角原子块 Cholesky。另报告 `V.active` 相对 `V.raw` metric 的差；实际 THC (Z=T^\dagger V T) 和核心计时使用 `V.active`，不能把 raw-(V) 的较大稠密代价当作生产加速基线。

- [ ] **Step 4: 运行测试**

```bash
PYTHONPATH=tools/exx_thc/src python3 -m unittest discover \
  -s tools/exx_thc/tests -p 'test_tt.py' -v
PYTHONPATH=tools/exx_thc/src python3 -m unittest discover \
  -s tools/exx_thc/tests -p 'test_metrics.py' -v
```

Expected: `OK`。

- [ ] **Step 5: 提交**

```bash
git add tools/exx_thc/src/exx_thc/tt.py tools/exx_thc/src/exx_thc/metrics.py \
        tools/exx_thc/tests/test_tt.py tools/exx_thc/tests/test_metrics.py
git commit -m "feat: add Coulomb-metric TT rank diagnostics"
```

## Task 7: 实现代数 occupied-THC 与不解压收缩

**Files:**
- Create: `tools/exx_thc/src/exx_thc/thc.py`
- Create: `tools/exx_thc/src/exx_thc/contract.py`
- Create: `tools/exx_thc/tests/test_thc.py`
- Create: `tools/exx_thc/tests/test_contract.py`

- [ ] **Step 1: 写已知秩、复数共轭和直接收缩失败测试**

```python
rng = np.random.default_rng(11)
t = rng.normal(size=(7, 3)) + 1j*rng.normal(size=(7, 3))
x = rng.normal(size=(5, 3)) + 1j*rng.normal(size=(5, 3))
y = rng.normal(size=(4, 3)) + 1j*rng.normal(size=(4, 3))
cbar = np.einsum("mx,ix,vx->miv", t.conj(), x, y)
raw = rng.normal(size=(7, 7)) + 1j*rng.normal(size=(7, 7))
v = raw @ raw.conj().T + 0.5*np.eye(7)
k_dense = dense_exchange(cbar, v)
k_thc = thc_exchange(t, x, y, v)
np.testing.assert_allclose(k_thc, k_dense, rtol=1e-11, atol=1e-11)
```

另测 `thc_exchange` 的最大临时 shape 不含 `(naux,nao,nocc)`，确保性能路径不构造 `cbar`。

- [ ] **Step 2: 实现确定性的复数 CP-ALS**

`thc.py` 使用 SVD 初始化、固定列归一化和相位约定；每次 ALS 用 Khatri--Rao 矩阵和带 ridge 行的 `numpy.linalg.lstsq` 更新三个因子。完整核心为：

```python
@dataclass(frozen=True)
class THCFactors:
    T: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    residual: float
    iterations: int
    converged: bool

def cp_als(cbar: np.ndarray, rank: int, max_iter: int = 100,
           rel_tol: float = 1e-8, ridge: float = 1e-12,
           seed: int = 0) -> THCFactors:
    if cbar.ndim != 3 or rank < 1:
        raise ValueError("cp_als expects a rank-3 tensor and rank >= 1")
    a = np.asarray(cbar, dtype=np.complex128, order="C")
    rng = np.random.default_rng(seed)

    def init_factor(unfold, rows):
        u, _, _ = scipy.linalg.svd(unfold, full_matrices=False,
                                   lapack_driver="gesdd")
        out = np.empty((rows, rank), dtype=np.complex128)
        copied = min(rank, u.shape[1])
        out[:, :copied] = u[:, :copied]
        if copied < rank:
            out[:, copied:] = (rng.normal(size=(rows, rank-copied))
                               + 1j*rng.normal(size=(rows, rank-copied))) / np.sqrt(rows)
        return out

    A = init_factor(a.reshape(a.shape[0], -1), a.shape[0])
    X = init_factor(a.transpose(1, 0, 2).reshape(a.shape[1], -1), a.shape[1])
    Y = init_factor(a.transpose(2, 0, 1).reshape(a.shape[2], -1), a.shape[2])

    def khatri_rao(left, right):
        return np.einsum("ir,jr->ijr", left, right).reshape(
            left.shape[0]*right.shape[0], rank)

    def solve_factor(unfold, kr):
        lhs = np.vstack((kr, np.sqrt(ridge)*np.eye(rank, dtype=np.complex128)))
        rhs = np.vstack((unfold.T,
                         np.zeros((rank, unfold.shape[0]), dtype=np.complex128)))
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0].T

    norm = max(float(np.linalg.norm(a)), np.finfo(float).tiny)
    previous = np.inf
    stable = 0
    residual = np.inf
    for iteration in range(1, max_iter+1):
        A = solve_factor(a.reshape(a.shape[0], -1), khatri_rao(X, Y))
        X = solve_factor(a.transpose(1, 0, 2).reshape(a.shape[1], -1),
                         khatri_rao(A, Y))
        Y = solve_factor(a.transpose(2, 0, 1).reshape(a.shape[2], -1),
                         khatri_rao(A, X))
        for column in range(rank):
            for factor in (A, X):
                scale = max(float(np.linalg.norm(factor[:, column])),
                            np.finfo(float).tiny)
                factor[:, column] /= scale
                Y[:, column] *= scale
            pivot = int(np.argmax(np.abs(A[:, column])))
            if abs(A[pivot, column]) > 0.0:
                phase = A[pivot, column] / abs(A[pivot, column])
                A[:, column] /= phase
                Y[:, column] *= phase
        rebuilt = np.einsum("mr,ir,vr->miv", A, X, Y)
        residual = float(np.linalg.norm(a-rebuilt) / norm)
        stable = stable + 1 if abs(previous-residual) < rel_tol else 0
        if stable >= 2:
            return THCFactors(A.conj(), X, Y, residual, iteration, True)
        previous = residual
    return THCFactors(A.conj(), X, Y, residual, max_iter, False)
```

每轮按 $A=T^*,X,Y$ 顺序更新；每列把范数吸收到 `Y`，并令 `A` 中最大绝对值元素为非负实相位。相对残差连续两轮变化 `<rel_tol` 才标记 `converged`。对每个 rank 用 seeds `0,1,2`，保留残差最低者，同时报告三次残差范围；不得只展示幸运 seed。

- [ ] **Step 3: 实现稠密与 THC 核心**

```python
def dense_exchange(cbar, v):
    return -np.einsum("miv,mn,nkv->ik", cbar, v, cbar.conj(), optimize=True)

def thc_exchange(t, x, y, v):
    z = t.conj().T @ v @ t
    s = y.T @ y.conj()
    return -x @ (z * s) @ x.conj().T
```

所有输入强制 `complex128`、C-contiguous；输出做 Hermiticity 诊断但不静默对称化。

- [ ] **Step 4: 运行测试**

```bash
PYTHONPATH=tools/exx_thc/src python3 -m unittest discover \
  -s tools/exx_thc/tests -p 'test_thc.py' -v
PYTHONPATH=tools/exx_thc/src python3 -m unittest discover \
  -s tools/exx_thc/tests -p 'test_contract.py' -v
```

Expected: 已知 rank-3 张量残差 `<1e-8`，直接与稠密收缩误差 `<1e-11`。

- [ ] **Step 5: 提交**

```bash
git add tools/exx_thc/src/exx_thc/thc.py tools/exx_thc/src/exx_thc/contract.py \
        tools/exx_thc/tests/test_thc.py tools/exx_thc/tests/test_contract.py
git commit -m "feat: add algebraic occupied-THC prototype"
```

## Task 8: 在 66 构建并取得小 Si 与 GaAs 参考快照

**Files:**
- Do not modify: remote `CMakeLists.txt`
- Create remotely: `/home/ghj/abacus/260810/mps-exx-k444/build/`
- Create remotely: `/home/ghj/abacus/260810/mps-exx-k444/runs/si_kp_pbe0/`
- Create remotely: `/home/ghj/abacus/260810/mps-exx-k444/runs/gaas_k444/`

- [ ] **Step 1: 执行版本硬门槛**

记录：本地 `master_ghj`、本地 feature HEAD、远端 feature HEAD、远端 LibRI include 路径及其 Git SHA（若该安装不是 Git tree，记录目录和所有关键头文件 SHA256）。输出必须形成：

```text
Execution: server=159.226.208.66, local_compute=no
ABACUS feature branch: codex/mps-exx-k444
Version verdict: feature-branch exception only when local and server HEAD strings match
LibRI verdict: source SHA, or SHA256 manifest when the installed headers are not a Git tree
Next action: build only after both verdicts have recorded evidence
```

用 `git rev-parse master_ghj`、`git rev-parse HEAD`、`git log -1 --oneline` 和 `sha256sum` 的实际输出替换说明文字；报告中不得保留尖括号变量。

任何 SHA 不匹配都先同步或重建，不使用旧二进制。

- [ ] **Step 2: 按 66 profile 配置独立 build**

在同一远程 shell 中加载 `~/.bashrc`，检查 `/home/ghj/abacus/260110/new/abacus-develop/CMakeLists.txt` 的已验证编译器/依赖路径。全部服务器路径通过 CMake cache 传入，保持远端源码工作区 clean：

```bash
cmake -S . -B /home/ghj/abacus/260810/mps-exx-k444/build \
  -DCMAKE_CXX_COMPILER=/home/apps/intel20u4/compilers_and_libraries_2020.4.304/linux/bin/intel64/icpc \
  -DMPI_CXX_COMPILER=/home/apps/intel20u4/compilers_and_libraries_2020.4.304/linux/mpi/intel64/bin/mpiicpc \
  -DCEREAL_DIR=/home/linpz/software/cereal/cereal-1.3.0 \
  -DCEREAL_INCLUDE_DIR=/home/linpz/software/cereal/cereal-1.3.0/include \
  -DELPA_LIBRARY=/home/linpz/software/elpa/elpa-openmp_2021.11.002/lib/libelpa_openmp.so \
  -DELPA_DIR=/home/linpz/software/elpa/elpa-openmp_2021.11.002 \
  -DELPA_INCLUDE_DIR=/home/linpz/software/elpa/elpa-openmp_2021.11.002/include/elpa_openmp-2021.11.002 \
  -DLIBRI_DIR=/home/ghj/abacus/250920/LibRI \
  -DLIBCOMM_DIR=/home/ghj/abacus/250920/LibComm \
  -DLibxc_DIR=/home/ghj/abacus/libxc-6.0.0-build \
  -DENABLE_MPI=ON -DENABLE_LCAO=ON -DENABLE_LIBRI=ON -DENABLE_LIBCOMM=ON \
  -DENABLE_MLALGO=OFF -DDEBUG_INFO=ON -DBUILD_TESTING=ON
cmake --build /home/ghj/abacus/260810/mps-exx-k444/build -j20
```

Expected: 构建成功并生成 `build/abacus_3p`；`CMakeCache.txt` 的 MPI/LCAO/LibRI/LibComm/DEBUG_INFO 为 `ON`、MLALGO 为 `OFF`；记录 executable 的绝对路径、`stat` 和 SHA256。配置后 `git status --porcelain` 仍为空；若 cache 参数不能复现参考 profile，则停止并记录第一个 configure 错误，不直接修改源码。

- [ ] **Step 3: 先跑小 Si 语义门槛**

在 `si_kp_pbe0` 副本中使用 1 rank、48 threads 和 dump 开关。要求 ABACUS 正常结束、快照齐全、Task 5 的原始 replay 和 occupied projection replay 都通过。未通过时停止，不提交 GaAs。

2026-08-11 的首次生产快照作业 `407502` 暴露了一个接口门槛：nspin=1 的 C/V/D/H 文件使用 EXXCMP1 `real64`，而最初的 replay/project 原型只接受 `complex128`。原始作业正常结束且 manifest 为 `session_state=complete`；不得通过改成复数算例绕过。新增回归要求 replay 按统一的输入标量 tag 分派到 `RI::Exx<double>` 或 `RI::Exx<complex<double>>`，Python compare/project 同样接受两种同型输入并保持输出类型。先在真实 server-66 LibRI 上通过 real64 直接结果测试，再继续本步骤的 raw/occupied replay。

- [ ] **Step 4: 运行 GaAs PBE 门槛**

在 GaAs run 目录复制 `INPUT_pbe` 为 `INPUT`，不设置 dump 开关，提交 1 MPI × 48 OMP。验收：SCF 正常收敛、最终 `drho < 1e-9`、无 `NaN/Inf`、82 NAO 和 64 k 点与预期一致。记录总时间、峰值内存、能量和作业号；PBE 只作为同 PP/NAO 的低成本输入门槛。

- [ ] **Step 5: 运行一次 EXX 快照**

复制 `INPUT_exx_snapshot` 为 `INPUT`，设置 dump，提交相同资源。验收：至少完成一次 `cal_exx_elec`、C/V/D/H/E 快照全部闭合、日志中记录 ABF 数量、块数、shape、`cal_Hs` 时间与峰值内存。该结果标记为“fixed-density EXX snapshot”，不称为收敛 PBE0。

- [ ] **Step 6: 保存快照清单且不删除结果**

对输入、日志、二进制、快照 manifest 和 summary 生成 `sha256sum`; 大文件留在 66，只把相对路径、大小和哈希写入本地 `tools/exx_thc/cases/gaas_k444/remote_artifacts.txt`。所有原始结果保留。

## Task 9: GaAs 零误差投影、TT 上界和 THC 精度扫描

**Files:**
- Modify: `tools/exx_thc/src/exx_thc/cli.py`
- Create: `tools/exx_thc/tests/test_scan.py`
- Create remotely: `runs/gaas_k444/analysis/scan.csv`
- Create remotely: `runs/gaas_k444/analysis/block_tail.csv`
- Create remotely: `runs/gaas_k444/analysis/factors/`

- [ ] **Step 1: 写 scan 输出 schema 的失败测试**

CSV 每行必须含：

```text
spin,k_index,q_index,block,method,threshold,rank_r1,rank_r2,rank_thc,
discarded_weight_1,discarded_weight_2,whitened_residual,cbar_rel_fro,
dense_bytes,factor_bytes,workspace_bytes,compression_ratio,
factor_seconds,seed_min_residual,seed_max_residual
```

测试要求缺任一列时 `validate_scan_table` 抛出异常。

- [ ] **Step 2: 先做 GaAs raw-(D) full reference 与零误差 occupied projection replay**

先把 `D.raw` 通过 replay 只做 periodization，得到 `D.full` 与 full-(D) (H,E) reference；再运行完整 k444 Fourier 往返、PSD 和 projector 检查，把 `C_occ` 与 `D.full` 作为收缩输入交给 `RI::Exx` replay，并继续把同一份原始 `D.raw` 作为 `D_post_path` 计算能量。硬门槛仍为 `H_rel_fro <=1e-10`、`E_abs_Ry_atom <=1e-10`。同时单列 active-(D) 生产结果与 full-(D) 的差。零误差门槛失败即停止，不能继续用局部 block 残差代替物理等价性。

- [ ] **Step 3: 扫描 TT 阈值**

对每个完整 Coulomb (q) 扇区白化后，扫描：

```text
1e-8 1e-7 1e-6 1e-5 1e-4 1e-3
```

保存两条 Schmidt 谱、键秩、discarded weight、实际误差与界。先计算理想 SVD/TT 存储上界；如果所有满足预期物理残差的点都低于 `10x` 压缩，直接写出 negative verdict，停止 CP/THC。

- [ ] **Step 4: 扫描代数 occupied-THC rank**

仅在 TT 上界通过时，对每个扇区从 `R=max(r1,r2)` 开始，以 `ceil(1.25*R)` 增长，直到白化残差达到阈值或因子字节数已使压缩比低于 `10x`。每个 rank 运行 seeds `0,1,2`、最多 100 ALS 轮。保存最优因子和全部 seed 残差，不保存解压后的稠密 `cbar`。

- [ ] **Step 5: 用原始 LibRI 重放检查 (H_x,E_x)**

对候选因子仅为精度验证而重构 `cbar_hat`，计算 `C_occ_hat=cbar_hat@O_pinv`，逆 Fourier 写 `C_candidate.active.exxcmp`，然后用 active-map replay 得到 LibRI 原始 (H,E)：`D_path` 使用与 reference 相同的 `D.full`，`D_post_path` 使用原始 `D.raw`。相对 (H) 误差在统一乘 (-2) 后不变；能量差先按现有 `post_process_Eexx` 乘自旋因子（`nspin=1` 时为 2），再换算：

```python
spin_factor = {1: 2.0, 2: 1.0, 4: 1.0}[nspin]
mev_per_atom = spin_factor * abs(delta_energy_ry) * 13.605693122994 * 1000 / natom
```

筛选 `H_rel_fro <= 1e-4` 且 `mev_per_atom <= 1.0` 的点；重构和 replay 时间不计入加速，只计入精度验证成本。

- [ ] **Step 6: 计算真实总存储与长尾**

`factor_bytes` 必须合计所有 (T,X,Y)、所有扇区 (Z)、rank/键/shape 索引；`workspace_bytes` 取直接收缩最大临时量。报告总体压缩比以及最差 10 个原子对/晶格块的残差。任何依赖少数异常块的长尾必须保留，不能只报平均值。

- [ ] **Step 7: 提交分析代码与结果摘要**

```bash
git add tools/exx_thc/src/exx_thc/cli.py tools/exx_thc/tests/test_scan.py \
        tools/exx_thc/cases/gaas_k444/remote_artifacts.txt
git commit -m "feat: scan GaAs occupied-product compression ranks"
```

## Task 10: 同节点不解压核心微基准

**Files:**
- Modify: `tools/exx_thc/src/exx_thc/contract.py`
- Modify: `tools/exx_thc/src/exx_thc/metrics.py`
- Create: `tools/exx_thc/tests/test_benchmark.py`
- Create remotely: `runs/gaas_k444/analysis/benchmark.json`

- [ ] **Step 1: 写计时协议测试**

`benchmark_kernel` 必须先执行 2 次不计时预热，再执行 7 次并报告 median、MAD、min、max；测试用 fake timer 验证调用次数为 9。计时区不得包含文件 I/O、CP-ALS 或 `cbar` 重构。

- [ ] **Step 2: 固定 66 的线程与节点**

在与参考 EXX 相同的 48-core、180000 MB 节点类型上运行 1 process × 48
BLAS/OpenMP threads。2026-08-11 的实际调度门槛要求 `ghj` 账户使用 640；
740 即使有空闲节点也不接受该账户：

```bash
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
```

用 `threadpoolctl.threadpool_info()` 写入 MKL/OpenMP 库和实际线程数。若稠密与 THC 路径线程数不同，整组计时作废。

- [ ] **Step 3: 对全部物理扇区分别测稠密与 THC**

稠密基线调用 `dense_exchange(cbar,V)`；THC 调用 `thc_exchange(T,X,Y,V)`，并通过内存监测确认 THC 路径没有 `(naux,nao,nocc)` 级临时量。每个扇区输出两条时间分布；总核心时间以各扇区 median 之和计算，而不是只挑最快 block。

- [ ] **Step 4: 验证数值和性能门槛**

每次计时前验证 `K_rel_fro <= 1e-4`。最终报告：

```text
core_speedup = sum(dense_median_seconds) / sum(thc_median_seconds)
amortized_5_speedup = 5 * dense_core / (factor_build + 5 * thc_core)
```

要求 `core_speedup >= 5`、`amortized_5_speedup >= 1`、压缩比 `>=10`。任一失败即分类为“只适合存储”或“代数低秩但不经济”，不进入 LibRI 生产实现。

- [ ] **Step 5: 提交基准代码和小型 JSON 摘要**

```bash
git add tools/exx_thc/src/exx_thc/contract.py tools/exx_thc/src/exx_thc/metrics.py \
        tools/exx_thc/tests/test_benchmark.py
git commit -m "bench: measure factorized occupied-THC contraction"
```

## Task 11: 编写并验证理论、测试和结果 TeX/PDF

**Files:**
- Create: `/Users/ghj/同步空间/AITP_project/exx_occupied_thc_feasibility/main.tex`
- Create: `/Users/ghj/同步空间/AITP_project/exx_occupied_thc_feasibility/Makefile`
- Create: `/Users/ghj/同步空间/AITP_project/exx_occupied_thc_feasibility/README.md`
- Create: `/Users/ghj/同步空间/AITP_project/exx_occupied_thc_feasibility/sections/01_problem.tex`
- Create: `/Users/ghj/同步空间/AITP_project/exx_occupied_thc_feasibility/sections/02_theory.tex`
- Create: `/Users/ghj/同步空间/AITP_project/exx_occupied_thc_feasibility/sections/03_method.tex`
- Create: `/Users/ghj/同步空间/AITP_project/exx_occupied_thc_feasibility/sections/04_results.tex`
- Create: `/Users/ghj/同步空间/AITP_project/exx_occupied_thc_feasibility/sections/05_conclusion.tex`
- Create: `/Users/ghj/同步空间/AITP_project/exx_occupied_thc_feasibility/appendices/provenance.tex`
- Create: `/Users/ghj/同步空间/AITP_project/exx_occupied_thc_feasibility/appendices/tests.tex`
- Create: `/Users/ghj/同步空间/AITP_project/exx_occupied_thc_feasibility/references.bib`

- [ ] **Step 1: 先检查 AITP 根目录下 2--3 个现有 TeX 工程的排版习惯**

只检查 `/Users/ghj/同步空间/AITP_project` 直接子目录，不使用 `Downloads` 或 staging 副本。记录复用的字体、页边距、表格和引用方式。

- [ ] **Step 2: 写理论章节，完整给出四层等式**

正文使用英文，`README.md` 可附中文执行摘要。`main.tex` 从以下可独立编译骨架开始，再按 Step 1 检查到的既有工程习惯调整排版而不改变章节职责：

```tex
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.4cm]{geometry}
\usepackage{fontspec,amsmath,amssymb,bm,booktabs,longtable,graphicx,hyperref}
\usepackage[numbers,sort&compress]{natbib}
\title{Feasibility of Occupied-Product Tensor Compression for RI Exact Exchange}
\author{Huanjing Gong}
\date{2026-08-10}
\begin{document}
\maketitle
\tableofcontents
\listoffigures
\listoftables
\input{sections/01_problem}
\input{sections/02_theory}
\input{sections/03_method}
\input{sections/04_results}
\input{sections/05_conclusion}
\appendix
\input{appendices/provenance}
\input{appendices/tests}
\bibliographystyle{unsrtnat}
\bibliography{references}
\end{document}
```

`02_theory.tex` 必须依次推导：原始 `CVCD`、(D=OO^\dagger) 的零误差变换、Coulomb 白化与 TT discarded weight、occupied-THC 直接收缩。明确复数约定

\[
\bar C_{\mu iv}=\sum_xT_{\mu x}^*X_{ix}Y_{vx},\quad
Z=T^\dagger VT,\quad S=Y^T Y^*,\quad K=-X(Z\odot S)X^\dagger.
\]

说明三阶 TT 只有两条键，DMRG 长链的指数状态空间优势不能直接移植；本研究测试的是数值低秩，而不是用术语推断加速。

`references.bib` 至少收录并在正文引用：Oseledets 2011 (`10.1137/090752286`)、Lu--Ying 2015 (`10.1016/j.jcp.2015.09.014`)、Lee--Lin--Head-Gordon 2020 (`10.1021/acs.jctc.9b00820`)、Qin et al. 2020 (`10.1021/acs.jpca.0c02826`) 和 Rettig--Lee--Head-Gordon 2023 (`10.1021/acs.jctc.3c00407`)。BibTeX 的作者、题名、卷页和 DOI 逐项核对，不从记忆补全缺失字段。

- [ ] **Step 3: 写方法、版本和硬门槛**

`03_method.tex` 写 GaAs PP/NAO/k 点、BvK Fourier、PSD/零模处理、阈值扫描、3 seeds、2 warmups+7 repeats、线程绑定和全量字节统计。`provenance.tex` 写服务器、作业号、commit、binary stat/hash、四个输入 hash 和远端大文件 manifest。

- [ ] **Step 4: 用脚本自动生成结果表和图**

从 `scan.csv` 与 `benchmark.json` 生成：Schmidt 谱；阈值--秩--压缩比；(H_x,E_x) 误差；稠密/THC 时间；因子构建摊销；最差 block 长尾。表格数字必须由归档数据生成，不手抄。

- [ ] **Step 5: 写结论，按五类之一明确判定**

只允许：`occupied-THC 值得进入生产接入`、`TT 低秩但 THC 不经济`、`只节省存储`、`核心可行但端到端尚未证明`、`占据乘积压缩不可行`。如果第一阶段通过，只能写“建议进入第二阶段”，不能提前声称 `cal_exx_elec` 已达到 (2\times)。

- [ ] **Step 6: 数值复现、编译和逐页检查**

```bash
make reproduce
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: 分析摘要与归档 SHA 一致；无 undefined references、LaTeX error 或未解释的 overfull box。把 PDF 每页渲染为 PNG，逐页检查公式、表格、图例、页码和跨页断行；修正后再次构建。交付 `main.pdf`，不是只交 TeX。

## Task 12: 最终验证、同步与条件式下一计划

**Files:**
- Modify: `docs/superpowers/specs/2026-08-10-mps-exx-tensor-compression-design.md`
- Modify: `tools/exx_thc/cases/gaas_k444/README.md`
- Create only if all first-stage gates pass: `docs/superpowers/plans/2026-08-10-exx-local-isdf-libri-integration.md`

- [ ] **Step 1: 运行本地完整验证**

```bash
git diff --check
PYTHONPATH=tools/exx_thc/src python3 -m unittest discover -s tools/exx_thc/tests -v
ctest --test-dir build -R 'MODULE_RI_exx_compression|08_EXX' --output-on-failure
```

Expected: `git diff --check` 无输出；Python `OK`；CTest `100% tests passed`。

- [ ] **Step 2: 审计默认行为和结果来源**

关闭 `ABACUS_EXX_COMPRESSION_DUMP` 重跑小 Si，确认数值和文件系统行为与基线相同。检查 GaAs 的输入、日志、job id、commit、binary hash、快照和分析文件哈希形成闭环。任何缺来源的数字从报告删除。

- [ ] **Step 3: 更新设计状态和最终 verdict**

把设计稿状态改为实际五分类之一，并列出通过/失败的每条门槛。失败结果与输入全部保留，不删除快照。

- [ ] **Step 4: 仅在同时通过 (10\times/5\times) 时写第二计划**

第二计划必须覆盖：真实轨道网格上的 local ISDF 插值点选择；(q)-dependent (Z^q)；LibRI factor 数据结构与 MPI 分布；不解压 `cal_Hs`；GaAs 带边 10 meV；`cal_exx_elec >=2x`；力/应力暂不进入。任何第一阶段门槛失败都不创建生产实现计划。

- [ ] **Step 5: 按 commit attribution 提交最终文档更新**

```bash
git add docs/superpowers tools/exx_thc
git commit -m "docs: report occupied-THC EXX feasibility"
git show -s --format=fuller HEAD
```

Expected: Author 为 `Codex <codex@openai.com>`，Committer 为 `AroundPeking <gonghuanjing@iphy.ac.cn>`。

- [ ] **Step 6: 安全同步到 66**

先确认两端 clean、remote 路径一致，再从本地推送到 `/home/ghj/git/abacus-develop-mps-exx.git`，远端使用旧 Git 兼容命令：

```bash
git push supercomputer codex/mps-exx-k444
ssh 159.226.208.66 'bash -lc '\''
source ~/.bashrc >/dev/null 2>&1 || true
cd /home/ghj/abacus/260810/mps-exx-k444/source
git pull --rebase origin codex/mps-exx-k444
git status --short --branch
git rev-parse HEAD
'\'''
```

Expected: 远端 HEAD 与本地相同且工作区 clean；不用 force push、reset 或 checkout 覆盖。

---

## 执行停止点

实施时每个硬门槛都是实际停止点：

1. 小 Si 原始重放不等价：修 dump/replay，不跑 GaAs。
2. 小 Si 或 GaAs 零误差占据投影不等价：停止并修正 BvK/LibRI 指标语义。
3. TT/SVD 理想上界达不到 (10\times)：停止 THC。
4. THC 精度下达不到 (10\times)：停止性能实现。
5. 直接核心达不到 (5\times) 或五次不能摊销：不做 LibRI 生产接入。
6. 第一阶段全部通过：只说明值得进入下一阶段，另写 local-ISDF/LibRI 计划后再测试 `cal_exx_elec` (2\times)。
