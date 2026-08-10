# GaAs k444 EXX 占据乘积空间张量压缩设计

日期：2026-08-10
状态：用户已批准，实施计划已完成，待执行
ABACUS 基线：`master_ghj@31ad8d2db354853249969e50a377881929eaf2fa`
开发分支：`codex/mps-exx-k444`
执行服务器：`159.226.208.66:62030`
远端工作区：`/home/ghj/abacus/260810/mps-exx-k444/source`

## 1. 研究问题与边界

本研究不再把“对每个三阶 \(C_{\mu ij}\) 块机械地做 MPS”作为主路线。DMRG 的有效类比是根据 Schmidt 谱和 discarded weight 压缩对物理量真正有用的状态空间。对 EXX，这个空间是占据轨道参与的轨道乘积空间，而不是 KS 波函数的原始 NAO 空间。

主路线固定为：

\[
D\text{ 的占据分解}
\;\longrightarrow\;
\text{Coulomb 度量化}
\;\longrightarrow\;
\text{局域 occupied-THC/ISDF}.
\]

逐块 TT/MPS 保留为对照，用来测量 Schmidt 谱、秩与截断阈值的关系，但不预设它是最终生产表示。

首轮只研究固定 PBE 密度矩阵上的一次 EXX 构造，不研究完整 PBE0 自洽、力、应力、\(\chi^0\) 或 GW。继续开发的最低目标为：

1. 轨道乘积表示或存储至少压缩 \(10\times\)；
2. 不解压的 `CVCD` 等价核心收缩至少加速 \(5\times\)；
3. ABACUS `cal_exx_elec` 至少加速 \(2\times\)；
4. 仍满足 EXX 能量、交换矩阵和带边精度门槛。

只减少文件大小、只减少内存，或先解压成稠密张量再调用原收缩，都不计为核心加速。

## 2. 从 `CVCD` 到占据乘积空间

ABACUS 按原子对和晶格矢量保存局域 RI 块

\[
C^{I,J\mathbf R}_{\mu ij}.
\]

块外层已经由原子局域性、实空间截断和稀疏映射处理，因此新表示必须保留这些块结构，不建立全局稠密张量。省略块标记后，EXX 的代表式为

\[
K_{ik}
=-
\sum_{jl\mu\nu}
C_{\mu ij}V_{\mu\nu}C_{\nu kl}^{*}D_{jl}.
\]

首先对密度矩阵做占据分解：

\[
D_{jl}=\sum_{v=1}^{N_{\rm occ}}O_{jv}O_{lv}^{*},
\qquad
O_{jv}=\sqrt{f_v}\,U_{jv}.
\]

定义

\[
\overline C_{\mu iv}
=
\sum_j C_{\mu ij}O_{jv},
\]

则有零近似误差的重写

\[
K_{ik}
=-
\sum_{\mu\nu v}
\overline C_{\mu iv}V_{\mu\nu}
\overline C_{\nu kv}^{*}.
\]

对零温绝缘体，这把第二个 AO 指标从 \(N_{\rm AO}\) 精确换成 \(N_{\rm occ}\)。对分数占据，根据密度矩阵本征值截断并单独报告占据截断误差。实施前必须检查 LibRI 当前收缩顺序是否已经等价利用了 \(D\) 的低秩。

对周期体系，占据分解必须在每个自旋和 \(\mathbf k\) 点的厄米非负 \(D^\sigma(\mathbf k)\) 上进行。不得对单个实空间 \(D(\mathbf R)\) 块独立做“占据分解”，因为该块未必是非负矩阵。投影完成后再映射回 LibRI 的原子对/晶格块布局。

ABACUS worktree 负责产生有来源的 \(C,V,D\)、占据轨道和参考 EXX；真正的不解压收缩最终位于 LibRI。只有秩谱、物理误差和理论浮点数同时通过门槛，才建立配套 LibRI worktree。

## 3. 压缩表示与直接收缩

### 3.1 Coulomb 度量下的 Schmidt/TT 对照

对 \(V=LL^\dagger\) 定义白化张量

\[
B_{Piv}
=
\sum_\mu L^\dagger_{P\mu}\overline C_{\mu iv}.
\]

上式中的白化必须对具有完整物理意义的 Coulomb 度量进行；不对任意的非对角 \(V^{IJ}\) 块分别 Cholesky。若 \(q=0\) 处存在零模或奇异校正，则在与基线相同的子空间内用本征分解或枢轴 Cholesky 定义 \(L\)。

对 \(B\) 做三阶 TT：

\[
B_{Piv}
\approx
\sum_{\alpha\beta}
G^{(1)}_{P\alpha}
G^{(2)}_{\alpha i\beta}
G^{(3)}_{\beta v}.
\]

对每个 TT 切分 \(p\)，定义 DMRG 式 discarded weight

\[
w_p(r_p)
=
\frac{\sum_{a>r_p}\sigma_{p,a}^{2}}
{\sum_a\sigma_{p,a}^{2}}.
\]

TT-SVD 的全局误差满足

\[
\|B-\widehat B\|_F^2
\le
\sum_p\sum_{a>r_p}\sigma_{p,a}^{2}.
\]

因此 TT 用于回答“真实数据的 Schmidt 秩是多少”。三阶张量只有两条键；若键秩不小，不得用 DMRG 的长链优势推断它会自然获得数量级收益。

### 3.2 主路线：local occupied-THC/ISDF

直接对占据投影后的张量建立可分离表示：

\[
\overline C_{\mu iv}
\approx
\sum_{x=1}^{R}
T_{\mu x}^{*}X_{ix}Y_{vx}.
\]

这里显式采用 \(T^*\) 的因子约定，使后续 Coulomb 内核在复数 \(k\) 点情况下仍是标准的 \(Z=T^\dagger VT\)。

在 ISDF 中，\(x\) 对应局域插值点 \(\mathbf r_x\)，并有

\[
X_{ix}=\phi_i(\mathbf r_x),
\qquad
Y_{vx}=\psi_v(\mathbf r_x),
\]

\[
\phi_i^*(\mathbf r)\psi_v(\mathbf r)
\approx
\sum_x
\phi_i^*(\mathbf r_x)\psi_v(\mathbf r_x)
\zeta_x(\mathbf r).
\]

定义

\[
Z_{xy}
=
\sum_{\mu\nu}
T_{\mu x}^{*}V_{\mu\nu}T_{\nu y},
\qquad
S_{xy}
=
\sum_vY_{vx}Y_{vy}^{*},
\]

则交换矩阵直接由

\[
\boxed{
K=-X\left[Z\odot S\right]X^\dagger
}
\]

构造，其中 \(\odot\) 是逐元素乘积。这个公式不恢复 \(\overline C\) 或 \(C\)，因此才是性能测试的生产候选。

对原子对块，\(T,X,Y,Z,S\) 都带原子和晶格块标记，插值点只从轨道支撑区交集中选取。不允许为方便原型而丢掉 LibRI 已有的局域性。

### 3.3 周期 \(k\) 点形式

对 \(\mathbf q=\mathbf k'-\mathbf k\)，使用晶体轨道的周期部分：

\[
u_i^{\mathbf k}(\mathbf r)^*
u_v^{\mathbf k+\mathbf q}(\mathbf r)
\approx
\sum_x
u_i^{\mathbf k}(\mathbf r_x)^*
u_v^{\mathbf k+\mathbf q}(\mathbf r_x)
\zeta_x^{\mathbf q}(\mathbf r).
\]

对应的 Coulomb 内核 \(Z^{\mathbf q}_{xy}\) 只依赖 \(\mathbf q\)，而不是独立依赖 \(\mathbf k,\mathbf k'\)。原型首先保持 LibRI 实空间块结构；只在数据证明 \(k\) 点卷积是瓶颈时，才考虑以 FFT 把 \(N_k^2\) 收缩变成近似 \(N_k\log N_k\)。

### 3.4 能量与算符两条压缩线

- `occupied-occupied` ISDF 只作为 EXX 能量的最大可压缩性上限，不单独通过交换矩阵验收。
- `AO-occupied` ISDF 是首个生产候选，它必须构造完整 \(H_x\) 并通过带边门槛。
- 后续若只需要特定带窗口，可研究 `occupied-target` 乘积空间，但它不代替首轮完整交换矩阵验证。

## 4. 阈值、误差传播与性能判据

压缩秩 \(R\) 由 Coulomb 度量下的残差而不是裸 \(C\) 的元素误差选择：

\[
\epsilon_R
=
\frac{\|B-\widehat B_R\|_F}{\|B\|_F}.
\]

对 \(G=B^\dagger B\) 有

\[
\|G-\widehat G\|_2
\le
\left(2\|B\|_2+\|\delta B\|_2\right)
\|\delta B\|_2,
\]

所以 Coulomb 白化残差比裸 \(C\) 残差更直接控制四中心积分和 EXX 误差。该界只用于解释趋势，最终仍以直接物理量验收：

\[
\frac{\|\widehat H_x-H_x\|_F}{\|H_x\|_F}\le10^{-4},
\qquad
|\widehat E_x-E_x|\le1\ {\rm meV/atom},
\]

并要求价带顶、导带底及带隙误差均不超过 \(10\) meV。

在简化全局维度 \(N=N_{\rm AO}\)、\(N_X=N_{\rm aux}\) 下，传统 RI 收缩约为

\[
O(N_XN^3),
\]

THC 直接收缩约为

\[
O(N^2R+NR^2),
\]

其理想浮点数比为

\[
S_{\rm ideal}
\sim
\frac{N_XN^2}{R(N+R)}.
\]

该式只是继续开发前的理论筛选；LibRI 的原子块稀疏、通信和因子构建成本必须进入实测。

压缩比必须统计全部 \(T,X,Y\)、所有 \(\mathbf q\) 的 \(Z^{\mathbf q}\)、索引和必要工作区；不允许只用某个因子的字节数声称 \(10\times\) 压缩。

所有截断点同时报告：

- 密度矩阵本征值和可选占据截断误差；
- TT 的 discarded weight、键秩与指标顺序；
- Coulomb 白化残差 \(\epsilon_R\) 与 ISDF 秩 \(R\)；
- 稠密、占据投影和 THC 表示的存储字节数；
- 理想浮点数、实测核心时间和因子构建时间；
- \(H_x\)、\(E_x\) 和带边误差。

## 5. 物理测试体系

固定使用两原子 GaAs 原胞：

- 轨道：Ga/As `8au_100Ry_3s3p3d2f`，每个原子 41 个 NAO，总计 82 个；
- \(k\) 点：Gamma-centered \(4\times4\times4\)；
- 辅助基：由相同 NAO 乘积和 `exx_pca_threshold=1e-4` 生成；
- 数据类型：实空间系数使用基线双精度类型，\(k\) 空间占据投影和收缩必须支持双精度复数；
- 参考路线：先完成同 PP/NAO 的 PBE 收敛门槛，再在固定 PBE 密度矩阵上构造一次 EXX；
- 对称性、Coulomb 方案、截断阈值和实空间范围在所有比较中保持不变。

旧 GaAs \(8^3\) 案例只提供参数和性能数量级参考。新 \(4^3\) 计算必须重新记录：

- PP、NAO 和输入文件的 SHA256；
- ABACUS 和后续 LibRI 的分支、提交和可执行文件路径；
- ABF 数目、\(C\) 块形状和块数；
- Slurm 作业号、节点、MPI/OMP 布局和环境；
- 参考 \(E_x\)、\(H_x\) 和带边输出。

## 6. 实验阶段

### 阶段 A：零误差占据投影与秩谱

1. 在未修改物理算法的基线 ABACUS 上完成 PBE 门槛。
2. 生成一次参考 EXX，并保存精确来源的 \(C,V,D,H_x,E_x\) 和占据轨道。
3. 验证 \(D=OO^\dagger\) 以及用 \(\overline C_{\mu iv}=\sum_jC_{\mu ij}O_{jv}\) 重建的 \(H_x,E_x\) 与原始 `CVCD` 在浮点精度内一致。
4. 检查 LibRI 现有收缩是否已经使用等价的占据低秩路径，防止重复计算收益。
5. 对 Coulomb 白化的 \(B_{Piv}\) 计算矩阵 SVD 上界、三阶 TT 的两组 Schmidt 谱和 discarded weight。
6. 扫描 \(10^{-8},10^{-7},10^{-6},10^{-5},10^{-4},10^{-3}\) 的白化残差或 discarded-weight 阈值。

阶段 A 的结果是“最佳无结构低秩上界”。若在预期物理误差区间内，连理想 SVD/TT 都无法达到 \(10\times\) 存储压缩或理论 \(5\times\) 收缩减量，则直接停止 THC 实现。

### 阶段 B：local occupied-THC/ISDF 精度

1. 从原子轨道支撑区交集中选择插值点，建立 `occupied-occupied` 和 `AO-occupied` 两条原型。
2. 以 Coulomb 白化残差为目标求解 \(T,X,Y\)，扫描直接秩 \(R\) 和相对阈值。
3. `occupied-occupied` 只报告 \(E_x\) 精度与最大可压缩性；`AO-occupied` 必须构造完整 \(H_x\)。
4. 允许先解压进入原始收缩来验证布局和精度，但该时间不计为加速。
5. 检查第 4 节的能量、矩阵和带边门槛。

阶段 B 的通过条件是：`AO-occupied` 表示同时满足全部物理门槛和至少 \(10\times\) 存储压缩。

### 阶段 C：因子化收缩微基准

1. 为通过阶段 B 的 `AO-occupied` 表示实现 \(K=-X[Z\odot S]X^\dagger\) 的不解压收缩。
2. 与原始 LibRI 稠密收缩逐元素比较输出。
3. 固定同一节点、线程绑定和输入，进行 2 次预热和至少 5 次计时。
4. 分别报告因子生成时间、单次收缩时间和包含因子生成的摊销时间。
5. 统计峰值内存和临时张量尺寸。

阶段 C 的通过条件是：

- 核心收缩加速至少 \(5\times\)；
- 因子生成成本在 5 次 EXX 更新内摊销；
- 因子存储至少比稠密 \(C\) 小 \(10\times\)，且峰值内存不高于稠密基线；
- 数值结果仍满足阶段 B 门槛。

### 阶段 D：ABACUS 端到端 EXX

把通过阶段 C 的收缩接入实验分支，比较同一 GaAs \(4^3\) 作业：

- `Exx_LRI::cal_exx_elec` 时间；
- 总 EXX 阶段时间；
- 总作业时间；
- 峰值内存；
- MPI 通信和序列化字节数；
- 参考物理量。

端到端成功门槛是 `cal_exx_elec` 至少加速 \(2\times\)。若核心达到 \(5\times\) 但 `cal_exx_elec` 低于 \(2\times\)，结论必须定位为“内核可行但当前 ABACUS/LibRI 端到端不值得集成”，并停止生产接入。

## 7. 性能测量协议

首轮使用 66 服务器单节点：

- 分区优先 `740`，资源形状 48 核、180 GB；若调度原因改用 `640`，必须重新记录节点型号；
- 1 MPI rank × 48 OpenMP threads 作为主要基线；
- 固定 `OMP_PROC_BIND`、`OMP_PLACES`、MKL 线程数和编译优化；
- 同一组对比尽量落在同一节点；
- Slurm 内存申请不超过节点实时可用内存，提交前重新查询；
- 不用文本 `Cs_data` 的写出时间计算压缩内核加速。

旧 \(8^3\) 案例表明完整文本 \(C,V\) 输出可以占据数分钟。新测试优先使用二进制诊断或进程内基准，并把 I/O 单独计时。

## 8. 软件边界

第一阶段在 ABACUS 分支中只加入诊断、数据导出和测试入口。密度矩阵占据分解、Coulomb 白化、TT 谱和 THC/ISDF 原型放入独立、无 MPI 假设的模块，避免直接嵌入 `Exx_LRI.hpp` 的控制流程。

计划中的职责边界为：

- ABACUS `source/source_lcao/module_ri/`：真实 \(C,V,D,O\) 获取、物理结果和计时接入；
- ABACUS 单元测试：占据投影、原子块布局、重构精度、阈值边界和确定性；
- 独立分析脚本：Schmidt 谱、discarded weight、ISDF 秩、误差表、压缩比和理论浮点数；
- LibRI 后续 worktree：占据低秩路径和不解压的 \(K=-X[Z\odot S]X^\dagger\) 收缩；
- 研究文档：理论推导、版本证据、测试表格、图和负面结果。

原型不得修改公共输入接口或默认计算路径。任何实验开关默认关闭，并在未进入生产实现前保持为明确的开发选项。

## 9. 测试要求

### 数学单元测试

- 人工构造幂等和分数占据密度矩阵，验证 \(D=OO^\dagger\) 及本征值截断；
- 人工构造已知 TT 秩为 \((r_1,r_2)\) 的 Coulomb 白化张量，验证 Schmidt 秩、discarded weight 和误差界；
- 人工构造已知 CP/THC 秩 \(R\) 的 \(\overline C_{\mu iv}\)，验证因子恢复和相位/尺度不定性；
- 零张量、秩一张量、满秩随机张量；
- 原子块和晶格块的往返布局；
- 固定阈值下结果可重复；
- 实数与后续复数接口的类型边界明确。

### 收缩单元测试

- 小尺寸随机正定 \(V\) 和厄米非负 \(D\)；
- 原始 `CVCD`、占据投影精确式和 THC 直接式逐元素比较；
- \(C=0\)、\(D=0\)、秩一 \(C\)；
- 相同因子的 \(K=-X[Z\odot S]X^\dagger\) 与显式 \(\widehat C\) 收缩一致；
- 性能路径不产生与原始 \(C\) 或 \(\overline C\) 同量级的隐式稠密临时量。

### 物理回归

- GaAs \(4^3\) 参考 EXX；
- \(E_x,H_x\) 和带边三重门槛；
- 原始未启用压缩的结果必须与基线保持数值一致；
- 串行和代表性多线程布局均通过。

## 10. 停止条件和结论分类

出现以下任一情况时，不继续主路径集成：

- 理想 SVD/TT 上界在物理门槛下都无法达到 \(10\times\) 存储压缩或理论 \(5\times\) 收缩减量；
- `AO-occupied` THC/ISDF 在物理门槛下压缩比低于 \(10\times\)；
- 因子化核心收缩实测加速低于 \(5\times\)；
- `cal_exx_elec` 实测加速低于 \(2\times\)；
- 压缩需要产生与原始 \(C\) 同量级的稠密临时量；
- 误差对不同原子对或晶格块表现出不可控长尾；
- 结果依赖非确定性 SVD 或线程顺序，无法稳定复现。

最终结论按以下方式分类：

1. **occupied-THC/ISDF 可行**：同时通过精度、\(10\times\) 压缩、\(5\times\) 核心和 \(2\times\) `cal_exx_elec` 门槛；
2. **TT 可压缩但 THC 不经济**：Schmidt 谱低秩，但可分离因子秩或因子构建代价过高；
3. **只适合节省存储**：精度和 \(10\times\) 压缩通过，但性能门槛未通过；
4. **局部内核可行但不值得集成**：核心达到 \(5\times\)，但 `cal_exx_elec` 低于 \(2\times\)；
5. **占据乘积表示不可行**：物理门槛下无法达到有意义的秩和性能。

## 11. 交付物

本研究完成后提供：

- 固定版本和输入的 GaAs \(4^3\) 参考数据；
- 密度矩阵占据分解的零误差对照；
- Coulomb 白化矩阵 SVD、TT Schmidt/discarded-weight 谱和 local occupied-THC/ISDF 秩谱；
- 核心与端到端性能表；
- 压缩比、误差和加速比图；
- 包含理论推导、实现边界、测试方法、正负结果的 TeX/PDF 研究文档；
- 若门槛通过，提供 ABACUS 与 LibRI 后续生产接入计划；
- 若门槛失败，保留全部输入、日志、诊断数据和失败原因，不删除结果。

## 12. 方法参考

- I. V. Oseledets, *Tensor-Train Decomposition*, SIAM J. Sci. Comput. 33, 2295 (2011), DOI: `10.1137/090752286`.
- J. Lu and L. Ying, *Compression of the electron repulsion integral tensor in tensor hypercontraction format with cubic scaling cost*, J. Comput. Phys. 302, 329 (2015), DOI: `10.1016/j.jcp.2015.09.014`.
- J. Lee, L. Lin, and M. Head-Gordon, *Systematically Improvable Tensor Hypercontraction: Interpolative Separable Density-Fitting for Molecules Applied to Exact Exchange, Second- and Third-Order Møller–Plesset Perturbation Theory*, J. Chem. Theory Comput. 16, 243 (2020), DOI: `10.1021/acs.jctc.9b00820`.
- X. Qin et al., *Interpolative Separable Density Fitting Decomposition for Accelerating Hartree–Fock Exchange Calculations within Numerical Atomic Orbitals*, J. Phys. Chem. A 124, 5664 (2020), DOI: `10.1021/acs.jpca.0c02826`.
- A. Rettig, J. Lee, and M. Head-Gordon, *Even Faster Exact Exchange for Solids via Tensor Hypercontraction*, J. Chem. Theory Comput. 19, 5773 (2023), DOI: `10.1021/acs.jctc.3c00407`.

## 13. 当前资源与版本记录

本设计建立时：

```text
Execution: server=159.226.208.66:62030, local_compute=no
ABACUS local master_ghj: 31ad8d2db354853249969e50a377881929eaf2fa
ABACUS branch: codex/mps-exx-k444
ABACUS server workdir: /home/ghj/abacus/260810/mps-exx-k444/source
Remote sync repository: /home/ghj/git/abacus-develop-mps-exx.git
Server Git: 1.8.3.1, using bundled git-new-workdir shared-workdir helper
Scheduler snapshot: partition 640 has 3 idle nodes; partition 740 has 5 idle nodes
Version verdict: source branch matches the recorded local master_ghj commit
Design verdict: approved by user for implementation planning
Next action: write a file-by-file implementation plan; do not start compute before plan review
```
