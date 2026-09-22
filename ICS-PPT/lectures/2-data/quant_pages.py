"""《专题：大模型量化的底层系统原理》的页面内容。

文案来自仓库根目录的 quant.md，共 26 页，除删去的十页外逐字照录，本文件不对其做任何改写。
排版上只做两件事：把 quant.md 的「页面标题」接到 p.title，把「屏幕核心文案」
按原有的层级接到 p.slide / p.table / p.code / p.demo；「讲师讲稿与互动」进
p.notes，「文献出处」进页脚。

例外有十五处，都按讲师要求改动，不在 quant.md 中，函数内有注释标明：
bandwidth_ledger 的硬件数据（DDR5-7200 与 RTX 5090，表中速度随之重算）；
bandwidth_ledger_cont 中的 RTX 5090 照片与算术强度对比；affine_grid 整页
（w、q、S、Z 的数轴图解）；zero_point_cost 与 zero_point_compute 两页重写
（从 affine_mapping 的公式推导交叉项，并比较零点带来的额外计算量）；symmetric_failure_grid
整页（真实权重上 Q4_0 与 Q4_1 两种格点的位置图）；symmetric_failure_cont 的控制台
输出（照录 quant_compare 的实际输出，Q4_0 与 Q4_1 两行加粗）；memory_wall 的对比表
（三条对比改为表格，补参数量与访存量，算术强度一列留空，数值读自论文图 3），以及
memory_wall_cont 页底从上一页移来的量化箭头图；granularity_spectrum 页底的实测表格（原续页的三条
数据，续页删去）；
alu_energy 右上角的台积电商标，以及该页表中的数据（统一为 45nm，面积改为台积电 45nm 的综合结果）；
fp32_myth 与 bandwidth_ledger 之间的 roofline_basics 整页（计算量、访存量、带宽与算术强度的定义与单位）；
alu_energy_cont 之后的 quant_overview 整页（量化把浮点数映射为整数、存储与计算两个环节、4 个权重的示例、收益与代价）；
fp32_myth 中「7 位有效十进制精度」的量级估算；
kquants_superblock 的拆分与补充（第一面只留工程痛点，页底补位宽-误差图；超块图移到续页，
二级量化两条改写为 d、dmin 与 sc、m 的关系，图中的还原公式随之改写；第二个续页补还原的两步
与 quant_compare 的三行对比表）；
核对 fp32_myth 至 alu_energy_cont 各页后修正的表述（fp32_myth 的系统结论，bandwidth_ledger_cont
的 BF16 限定与「近似成反比」，memory_wall 与 memory_wall_cont 中 BERT 的并行方式、token 与
1/4 的前提，alu_energy_cont 的面积比值）。

另有一组按仓库根目录 ERRORS.md 的审校意见（第 11–16 条）修正的表述，函数内同样有注释标明：
fp32_myth 中「完全钝化」等绝对化结论（改为实验观察并注明限度，页脚补大模型量化的出处）；
bandwidth_ledger 的表头与估算前提（权重数据量、每 token 读一遍权重的上界）；memory_wall 的
测量平台与 memory_wall_cont 的「满负荷运转」「不改变计算图」；affine_mapping 的整数零点与
浮点偏移、误差界的前提；zero_point_compute 中 ③④ 离线计算的条件；symmetric_failure 的
RMSNorm、Q4_0 的格点范围与「100% 覆盖」；ai_float_formats 中 FP8 两种格式的用途；
nf4_lut 的正态假设与近似等概率；lab_release 按实验题面（Qwen3-VL-2B，A–D 四个部分）重写。
修改前的版本保存在同目录的 quant_pages.py.orig。

此后对全文的知识点又做了一次审校，修正的表述同样在函数内注明「按全文审校」：fp32_myth 的
「设计初衷」与参数量级；memory_wall 与 memory_wall_cont 讲稿中时延的归一化基准与 117 ~ 266
的来源；affine_mapping 讲稿中「全部技巧」一句；symmetric_failure 标明步长按整个张量计算；
clipping_tradeoff 的比例、阈值、权衡目标与页脚标题；granularity_spectrum 的 $M$、公式分子、
表头（实测的是 ffn_down 的前 4096 个权重）与页脚文献；kquants_superblock_cont 讲稿的适用范围；
kquants_superblock_cont2 页脚的作者与标题；zero_point_cost 与 kquants_superblock 讲稿中写死的页码；K_SCALES_DECODE 的首行注释；nf4_lut 还原时乘以
块的缩放系数；lab_release 讲稿中的部分编号；topic_cover 的「整数溢出」与 topic_cover_cont 的第 3 个核心问题。
这次审校之前的版本保存在 quant_pages.py.pass1，最后两处改动之前的版本保存在 quant_pages.py.pass2，
插入 quant_overview 之前的版本保存在 quant_pages.py.pass3。
quant.md 第 10 页（KL 散度标定算法）、第 16–20 页（第 4 幕「底层硬件算术与位级并行技术」）、
第 15 页（混合精度配方）、第 23 页（Cache 实测）、第 24 页（PTQ 流水线）与第 26 页（结语）
按讲师要求删去，没有对应的函数；
topic_cover_cont 的第 4 个核心问题（无分支解包与通道隔离加法）只在第 4 幕回答，随之删去。

一页放不下时拆成同名的两页（`*_cont`，kquants_superblock 按讲师要求拆成三页）——lecturekit 会把同名页合并为提纲里的
一行、共用一个页码，因此拆页不影响编号，也不需要改动文案。

排版约定沿用 pages.py：粗体后面不跟全角冒号，`$$` 公式块整体缩进一格，
含反斜杠的文案一律写成 Python 原始字符串。
"""


K_SCALES_DECODE = """// 提取子块 0 的 6 位步长 sc
uint8_t sc0 = scales[0] & 63;                     // 取第 0 字节低 6 位
// 提取子块 4 的跨字节步长（低 4 位拼高 2 位）：
uint8_t sc4 = (scales[8] & 0x0f) | ((scales[0] >> 6) << 4);"""

CODEX_CREDIT = "配图由 OpenAI Codex 图像生成模型绘制，仅作概念示意。"



def topic_cover(p):
    p.title("大模型量化的底层系统原理")
    p.image("assets/quant-cover.jpg", width_px=700).footnote(CODEX_CREDIT)
    p.slide("""
- **主讲内容**：大模型量化——体系结构瓶颈、定点映射与底层位运算
- **核心视角**：微架构能耗、冯·诺依曼访存墙与字节打包
""")
    p.notes("""
同学们，前面我们花了两周时间攻克了计算机系统底层的数据表示：补码、溢出、阶码、尾数。
大家可能觉得这只是教科书里的枯燥规范。今天这堂课，我们要把视角拉到当今全球计算技术的
最前沿——大模型量化。我们将看到，大家刚刚学过的每一个 bit、每一次移位、每一次掩码，
是如何在真实工业系统里支撑起千亿参数大模型飞速运转的。
""")


def topic_cover_cont(p):
    p.title("大模型量化的底层系统原理")
    p.slide("""
- **核心问题**：
  1. 为什么当代大模型系统要主动“抛弃”IEEE 754 单精度浮点数？
  2. 大模型自回归推理中，内存带宽是如何以线性关系卡死生成速度的？
  3. 工业级 GGUF 规范（Q4_0 / Q4_1 / Q4_K）如何在字节中排布量化块？Q4_K 又如何用二级量化压缩元数据开销？
""")


def fp32_myth(p):
    p.title("破局：高精度神话的破灭")
    p.slide(r"""
- **IEEE 754 FP32 的历史使命**：
  - 23 位尾数带来约 7 位有效十进制精度：$2^{23} \approx 8.4 \times 10^{6}$，量级为 $10^{7}$。
  - **设计初衷**：为数值计算统一浮点格式与舍入规则；误差要求更严的计算通常用 FP64。
- **大模型的参数与表征本质**：
  - 现代大模型是典型的**超大规模过参数化系统**，参数量从十亿级到万亿级。
  - **自注意力机制的核心逻辑**：推理结果主要取决于 Softmax 归一化后注意力分布的相对大小，以及激活通道之间的相对强弱。
  - **实验观察**：采用合适的量化方法，许多大模型的权重降到 8 位或 4 位后精度只小幅下降，但量化误差仍会改变 logit 与生成结果。
- **系统结论**：在大模型时代强行沿用传统科学计算的高能耗、大位宽 FP32，不仅在算力和显存上存在极大浪费，还会成倍增加自回归推理需要搬运的字节数，加剧“显存带宽”的限制。
""").footnote("文献出处：Goldberg, D. *What Every Computer Scientist Should Know About "
              "Floating-Point Arithmetic* (1991)；Dettmers, T. et al. *LLM.int8()*, NeurIPS (2022)；"
              "Frantar, E. et al. *GPTQ*, ICLR (2023).")
    p.notes(r"""
科学计算中的长时间积分会累积舍入误差，因此需要 FP32 甚至 FP64。大模型推理的情况不同：
实验表明，采用合适的量化方法，权重降到 8 位（LLM.int8()）或 4 位（GPTQ）后，困惑度与
任务精度只小幅下降。这是实验结果，不是普遍定理：LLM.int8() 发现 6.7B 以上的模型会出现
少数数值很大的离群特征，直接按 8 位量化精度明显下降，需要把这些维度单独留在 16 位计算。
因此能否降低精度、降到多少位，要用具体的模型、层与任务检验。推理时若沿用 FP32 权重，
每个 token 搬运的字节数是 BF16 的两倍。

7 位是量级估算：23 位二进制尾数能区分 $2^{23} \approx 8.4 \times 10^{6}$ 个值，与 7 位十进制数能区分的
$10^{7}$ 个值同一量级。换一个角度，相邻两个 FP32 数的相对间距约为 $2^{-23} \approx 1.2 \times 10^{-7}$，
误差出现在第 7 位十进制数字附近。算上隐含的首位 1，有效位是 24 位，$\log_{10} 2^{24} \approx 7.2$，结论相同。
""")


def roofline_basics(p):
    p.title("基本量：计算量、访存量、带宽与算术强度")
    p.table(
        headers=["物理量", "符号", "含义", "单位"],
        rows=[
            ["**计算量**", "$W$", "一次任务执行的浮点运算次数，1 次乘加计 2 次", "FLOP"],
            ["**访存量**", "$Q$", "一次任务在内存与芯片之间传输的字节数", "Byte"],
            ["**峰值算力**", "$P$", "芯片每秒最多执行的浮点运算次数", "FLOP/s"],
            ["**带宽**", "$B$", "内存每秒最多向芯片传输的字节数", "Byte/s"],
            ["**算术强度**", "$I = W / Q$", "每传输 1 字节数据执行的运算次数", "FLOP/Byte"],
        ],
        align=["left", "left", "left", "left"],
    )
    p.slide(r"""
- **单位前缀**：G = $10^9$，T = $10^{12}$。
- **写法提示**：GFLOPs 中的小写 s 表示复数，指运算次数；TFLOPS 中的大写 S 表示每秒，指运算速率。
""")
    p.notes("""
这一页把后面反复出现的五个量放在一起。计算量和访存量描述一次任务，峰值算力和带宽描述硬件。
算术强度是计算量与访存量的比值：权重的位宽越小，访存量越小，算术强度越高。
1 次乘加包含 1 次乘法和 1 次加法，计 2 次浮点运算。
""")


def bandwidth_ledger(p):
    p.title("硬件基础：内存带宽决定生成速度上限")
    p.slide(r"""
- **硬件现实基准**：
  - **PC 内存**：双通道 DDR5-7200 峰值带宽 = $7200\text{ MT/s} \times 8\text{ B} \times 2 = \mathbf{115.2\text{ GB/s}}$
  - **消费级显卡**：NVIDIA RTX 5090 显存峰值带宽 = $\mathbf{1792\text{ GB/s}}$
- **以 1.24B 小模型（Llama-3.2-1B）单请求自回归推理为例**：
  - 速度上限 = 峰值带宽 ÷ 权重数据量，前提是每个 token 恰好读取一遍全部权重；KV cache、激活、反量化与内核效率都未计入，实测速度低于此值。
""")
    p.table(
        headers=["量化格式与位宽", "模型大小（按平均位宽计）", "DDR5-7200 理论速度上限",
                 "RTX 5090 理论速度上限"],
        rows=[
            ["**BF16 (16 位)**", "2.47 GB", "47 token/s", "725 token/s"],
            ["**Q8_0 (8.5 位)**", "1.31 GB", "88 token/s", "1365 token/s"],
            ["**Q4_K (4.5 位)**", "**0.70 GB**", "**166 token/s**", "**2578 token/s**"],
        ],
        align=["left", "right", "right", "right"],
    )
    p.notes("""
请大家关注这张对比表：同一个 12 亿参数的模型，在标准的双通道 DDR5 内存环境下，若采用
16 位浮点格式，单请求生成速率上限仅为 47 token/s；而一旦压缩至 4 位量化格式，理论吞吐
上限将跃升至 166 token/s。表中的上限只由带宽与权重字节数决定，实际运行中还要读取 KV cache
与激活、执行反量化，内核也达不到峰值带宽，因此实测速度低于表中的值。
""")
    p.cite(title="Intel Announces New Intel Core Ultra 200S Plus Series Desktop Processors",
           author="Intel", year="2026", venue="Intel Newsroom, March 11, 2026",
           url="https://www.intel.com/content/www/us/en/newsroom/news/client-computing/"
               "intel-announces-new-intel-core-ultra-200s-plus-series-desktop-processors.html",
           key="intel-200s-plus")


def bandwidth_ledger_cont(p):
    p.title("硬件基础：内存带宽决定生成速度上限")
    p.image("assets/ext/rtx5090.jpg", width_px=420).footnote(
        "GeForce RTX 5090 Founders Edition。照片来自 Wikimedia Commons，"
        "ZMASLO 视频截帧，CC BY 3.0，已裁剪。")
    p.slide(r"""
- **硬件铁律**：
  - 单请求推理场景下，每个权重从内存搬进芯片只参与 1 次乘加计算，算术强度极低（BF16 权重：$2\text{ FLOP} \div 2\text{ Byte}$，约为 $1\text{ FLOP/Byte}$）。
  - **对比 RTX 5090**：BF16 张量峰值算力 209.5 TFLOPS ÷ 显存带宽 1792 GB/s ≈ 117 FLOP/Byte，算术强度达到该值时计算单元才能满载。单请求推理的算术强度只有它的 1/117，计算单元的利用率上限不足 1%。
  - **模型的生成吞吐量上限与权重的物理位宽近似成反比**
""").footnote("文献出处：Llama-3.2-1B-Instruct Safetensors Spec & JEDEC DDR5 Standard.")
    p.cite(title="NVIDIA RTX Blackwell GPU Architecture", author="NVIDIA",
           year="2025", venue="V1.1, Table 3",
           url="https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/"
               "nvidia-rtx-blackwell-gpu-architecture.pdf", key="rtx-blackwell")


def memory_wall(p):
    p.title("问题实测：自回归推理深陷访存受限")
    p.image("assets/roofline-latency.svg", width_px=660)
    p.slide("**IEEE Micro 2024 实测：BERT-Base vs GPT-2**（序列长度 4096，CPU：Xeon Gold 6242）")
    p.table(
        headers=["模型", "参数量", "计算量", "访存量", "算术强度 (FLOP/Byte)", "端到端相对时延"],
        rows=[
            ["**BERT-Base**", "110M", "1324 GFLOPs", "11.2 GB", "", "84"],
            ["**GPT-2**", "124M", "1012 GFLOPs", "507.8 GB", "", "2344"],
        ],
        align=["left", "right", "right", "right", "center", "right"],
    )
    p.notes("""
计算量相近的模型，端到端延迟却可能存在数十倍的差距。实测数据清晰地揭示了这一现象的根源：
BERT 作为 Encoder 架构，整段序列在一次前向传播中并行处理，每个权重被序列中的全部 token 复用，算术强度超过 100 FLOP/Byte；
而大语言模型在逐 Token 自回归生成时，每次前向传播仅处理一个新 Token，却必须完整遍历全部
权重，算术强度骤降至 2 FLOP/Byte。在这种访存严重受限的场景下，缩减权重数据位宽是提升系统
吞吐最直接有效的途径。

表中二者的参数量与计算量相近，访存量相差约 45 倍，端到端时延相差约 28 倍。时延是相对值，
论文以 BERT-Base 处理长度为 128 的序列所用的时间为 1。算术强度一列留空，
请学生用计算量除以访存量算出，再对照左图中两者的位置：BERT-Base 为 1324 ÷ 11.2 ≈ 118
（论文图中标注 117，差在访存量的舍入），GPT-2 为 1012 ÷ 507.8 ≈ 2。续页的「根源剖析」给出这两个值。
""")


def memory_wall_cont(p):
    p.title("问题实测：自回归推理深陷访存受限")
    p.slide(r"""
- **根源剖析**（以 RTX 5090 的平衡点约 117 FLOP/Byte 为参照）：
  - **BERT（整段序列一次前向传播）**：算术强度高达 **117 ~ 266 FLOP/Byte**
    $\implies$ 达到或超过平衡点，处于**算力受限区**，性能上限由峰值算力决定。
  - **GPT-2（自回归逐 token 生成）**：算术强度跌落至仅 **2 FLOP/Byte**
    $\implies$ 处于**访存受限区**，性能上限由带宽决定，计算单元大部分时间在等待数据。
- **量化的作用**：模型结构不变，运行时换用带反量化的算子；权重数据量降至约 $1/4$（BF16 → 4 位），工作点在 Roofline 上右移，算术强度约提高 4 倍。
""").footnote("文献出处：Gholami, A. et al. *AI and Memory Wall*, IEEE Micro (2024).")
    p.image("assets/roofline-shift.svg",
            alt="RTX 5090 的 Roofline 上，单请求生成的工作点从 BF16 的 1 FLOP/Byte 经 8 位的 2 "
                "移到 4 位的 4，仍在平衡点 117 左侧的访存受限区",
            width_px=760)
    p.notes("""
算术强度只决定工作点在 Roofline 上的位置，也就是性能的上限；实际能达到多少，还取决于内核的
实现，要靠实测。上一页的时延是论文在 Intel Xeon Gold 6242 CPU 上测得的，这里借 RTX 5090 的
Roofline 说明两类负载的区别，两者不是同一台机器。

117 ~ 266 取自论文图 3(c)：BERT-Base 与 BERT-Large 在序列长度 128 至 4096 上的算术强度，
最小值 117 是 BERT-Base 在 4096 上的值，最大值 266 是 BERT-Large 在 512 上的值。
""")


def alu_energy(p):
    p.title("微架构视角：晶体管能耗与芯片面积核算")
    p.slide(r"""
- **45nm 工艺标准运算单元测试数据（Horowitz 经典基准）**：
""").image_right("assets/ext/tsmc-wordmark.svg", alt="台积电商标", width_px=150).footnote(
        "台积电（TSMC）商标。图片来自 Wikimedia Commons（File:TSMC wordmark.svg），"
        "公有领域，商标权归台积电所有。")
    p.table(
        headers=["运算单元类型", r"硅片物理面积 ($\mu m^2$)", "每次运算能耗 (pJ)",
                 "相对能耗倍数"],
        rows=[
            ["**FP32 乘法**", "7700", "3.7 pJ", r"**$123\times$**"],
            ["**FP32 加法**", "4184", "0.9 pJ", r"$30\times$"],
            ["**FP16 乘法**", "1640", "1.1 pJ", r"$37\times$"],
            ["**INT8 乘法**", "**282**", "**0.2 pJ**", r"**$6.7\times$**"],
            ["**INT8 加法**", "**36**", "**0.03 pJ**", r"**$1.0\times$ (基准)**"],
        ],
        align=["left", "right", "right", "right"],
    )
    p.notes("""
请大家看表里的相对能耗倍数：做 1 次 FP32 乘法的电能，足以支撑 120 多次 8 位整数加法！
这就解释了为什么当今体系结构设计者在布局张量核心与 NPU 时，会将成千上万个紧凑的定点 ALU
集成于硅片之中。每一位位宽的缩减，换来的都是微架构能效比与吞吐密度的飞跃。
""")
    p.cite(title="Efficient Methods and Hardware for Deep Learning", author="Han, S.",
           year="2017", venue="Stanford CS231n, Lecture 15",
           url="https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture15.pdf",
           key="han-cs231n")


def alu_energy_cont(p):
    p.title("微架构视角：晶体管能耗与芯片面积核算")
    p.image("assets/alu-floorplan.jpg", width_px=500).footnote(CODEX_CREDIT)
    p.slide(r"""
- **门电路级的物理开销**：
  - 32 位浮点加乘法器需要容纳庞大且繁杂的计算与判定电路。
  - 8 位整数加法器只需一条进位链，面积约为 FP32 加法器的 1/116（36 vs 4184 $\mu m^2$）。
  - 8 位整数乘法器的面积约为 FP32 乘法器的 1/27（282 vs 7700 $\mu m^2$）。
""").footnote("文献出处：Horowitz, M. *1.1 Computing's energy problem (and what we can do "
              "about it)*, ISSCC (2014)；面积：Han, S., Stanford CS231n (2017).")


def quant_overview(p):
    p.title("量化概览：把浮点数映射为整数")
    p.slide("""
- **量化**：把模型中的浮点数（BF16 / FP32）映射为少量位的整数，例如 8 位或 4 位整数。
- **存储**：模型文件保存映射后的整数，每个权重占用的位数随之减少。
- **计算**：推理时把整数映射回浮点数的近似值，再参与计算。
- **示例**：4 个权重按同一比例映射为 −7 ~ 7 的 4 位整数。
""")
    p.table(
        headers=["**浮点数**", "0.12", "−0.53", "0.97", "−0.08"],
        rows=[
            ["**4 位整数**", "1", "−4", "7", "−1"],
            ["**映射回的近似值**", "0.139", "−0.554", "0.970", "−0.139"],
        ],
        align=["left", "right", "right", "right", "right"],
    )
    p.slide("""
- **收益**：BF16 → 4 位时权重数据量约为 1/4，带宽受限的生成速度上限随之提高；激活也量化时，乘加可以由能耗与面积更小的整数运算单元完成。
- **代价**：映射回的值是近似值，与原来的浮点数之间有误差。
""")
    p.notes("""
量化做的事情是把浮点数映射为整数。模型文件里存的是映射后的 8 位或 4 位整数，
推理时再把这些整数映射回浮点数的近似值，参与计算。

示例中 4 个权重按同一比例映射到 −7 到 7：绝对值最大的 0.97 对应 7，其余的数按同一比例换算后
取最近的整数，−0.53 对应 −4。把 −4 映射回浮点数得到 −0.554，与原值相差 0.024；
−0.08 映射回 −0.139，相差约 0.059。这些差值就是量化带来的误差。

整数的位数越少，能表示的整数个数越少，数据量越小，误差也越大。
""")



def affine_mapping(p):
    p.title("形式化：均匀仿射量化映射")
    p.slide(r"""
- **映射任务**：将连续浮点区间 $[w_{\min}, w_{\max}]$ 投射至 $b$-bit 整数集合 $[q_{\min}, q_{\max}]$。
- **量化编码公式**：
 $$q = \text{clip}\left(\left\lfloor \frac{w}{S} \right\rceil + Z, \; q_{\min}, \; q_{\max}\right)$$
- **反量化重构公式**：
 $$\hat{w} = S \times (q - Z)$$
- **参数的物理意义**：
  - **步长 $S$**：正浮点数，表示离散网格的单位刻度间距：$S = \frac{w_{\max} - w_{\min}}{q_{\max} - q_{\min}}$。
  - **零点 $Z$**：实数 0 对应的整数码，$Z = q_{\min} - \left\lfloor w_{\min} / S \right\rceil$，使 $w_{\min}$ 映射到 $q_{\min}$。整数量化方案规定 $Z$ 为整数，此时 $q = Z$ 反量化得到的 $\hat{w}$ 恰好为 0。
""").footnote("文献出处：Jacob, B. et al. CVPR (2018).")
    p.notes("""
这两个公式是均匀量化的基础。$S$ 决定尺子的刻度有多细，它本身是一个浮点数；$Z$ 是把
原点固定在哪个整数格上。只要 $w$ 没有超出量化范围、$S$ 与 $Z$ 本身没有舍入，四舍五入带来的
误差不超过半个刻度。GGUF 的 Q4_1 不存 $Z$，而是把 $-S Z$ 合成一个 FP16 偏移 $m$ 直接存储，
$m$ 不必是步长的整数倍，因此 0.0 不一定落在格点上。
后面讲的非对称量化、分组与截断，都是在控制这个刻度 $S$ 的大小；截断缩小 $S$ 的代价是
范围外的值产生截断误差。
""")


def affine_grid(p):
    p.title(r"图解：$w$、$q$、$S$、$Z$ 在数轴上的含义")
    p.image("assets/affine-grid.svg", width_px=1100)


def zero_point_cost(p):
    p.title("零点 $Z$ 引发的交叉项惩罚")
    p.slide(r"""
- **由反量化公式**：$\hat{w} = S\,(q - Z)$。对称量化令 $Z = 0$，公式退化为 $\hat{w} = S\,q$。
- **矩阵乘法 $Y = XW$ 的每个输出元素是一个点积**：$X$ 的一行与 $W$ 的一列逐项相乘再求和，$y = \sum_{k=1}^{K} x_k w_k$，其中 $K$ 是向量长度。
- **激活与权重各自量化**，各有一组步长与零点：
  - 激活：$x_k \approx S_x\,(a_k - Z_x)$，整数码 $a_k$ 随每次输入变化。
  - 权重：$w_k \approx S_w\,(b_k - Z_w)$，整数码 $b_k$ 在模型离线量化之后固定不变。
- **代入点积，按乘法分配律展开**：
 $$y \approx S_x S_w \sum_{k=1}^{K} (a_k - Z_x)(b_k - Z_w) = S_x S_w \Big[\, \underbrace{\sum a_k b_k}_{\text{①}} - \underbrace{Z_w \sum a_k}_{\text{②}} - \underbrace{Z_x \sum b_k}_{\text{③}} + \underbrace{K Z_x Z_w}_{\text{④}} \,\Big]$$
- ② 与 ③ 由一侧的零点乘以另一侧整数码之和得到，称为**交叉项**。
""")
    p.notes(r"""
「形式化：均匀仿射量化映射」一页的反量化公式同时用于激活和权重。矩阵乘法的每个输出元素都是一个点积，把两个反量化
公式代入点积，再用乘法分配律展开，得到四项：① 是整数码的点积；② 和 ③ 各含一个零点，
是交叉项；④ 是常数。对称量化时两个零点都是 0，后三项全部消失。
""")


def zero_point_compute(p):
    p.title("零点 $Z$ 带来的额外计算量")
    p.slide(r"""
- **展开式中各项的计算时机**：
  - ① $\sum a_k b_k$：$K$ 次 8 位整数乘加，对称量化同样需要。
  - ② $Z_w \sum a_k$：$a_k$ 随输入变化，推理时现算 $\sum a_k$，多 $K$ 次加法。
  -   - ③ $Z_x \sum b_k$ 与 ④ $K Z_x Z_w$：$\sum b_k$ 只含权重，通常情况下 $Z_x$ 与 $Z_w$ 在推理前确定，此时两项都可以离线算好，推理时作为常数加上。
""")
    p.table(
        headers=["计算方式", "计算一次点积的惩罚", "能否用 8 位整数乘法"],
        rows=[
            ["对称量化", "无", "能"],
            ["非对称，按上页展开", r"求 $\sum a_k$ 的 $K$ 次加法", "能"],
        ],
        align=["left", "left", "left"],
    )
    p.slide(r"""
- **对称量化令 $Z_x = Z_w = 0$，②③④ 全部为 0，只剩纯整数点积 ①。**
""").footnote("文献出处：Jacob, B. et al. CVPR (2018); Krishnamoorthi, R. *Quantizing deep "
              "convolutional networks for efficient inference: A whitepaper* (2018).")
    p.notes(r"""
逐项先减零点再相乘的做法最直接，但 8 位码减去零点之后取值范围是 $[-255, 255]$，需要
9 位，只能改用 16 位乘法，每条指令完成的乘加数减半。展开之后，① 仍是纯 8 位整数点积，零点带来的工作集中在 ②：推理时对激活的整数码求和。
这个和对同一行激活只需求一次，由该行的所有输出列共用。以 Llama-3.2-1B 的一个
2048 × 2048 投影矩阵、生成 1 个 token 为例，① 约 419 万次乘加，② 为 2048 次加法。

GGUF 的 Q4_1 把反量化写作 $\hat{w} = d\,q + m$，$m$ 是浮点数，对应这里的 $-S_w Z_w$ 一项。llama.cpp 把激活
量化为 Q8_1 时，每 32 个数一块，块内同时存下 $d \cdot \sum a_k$，点积时每块只多 1 次乘加。
""")


def symmetric_failure(p):
    p.title("对称量化的不足：数据非对称分布下的精度雪崩")
    p.slide(r"""
- Llama-3.2 最终 RMSNorm（`model.norm.weight`）共 2048 个参数，**取值全部为正**，分布在 $[+0.0417, +2.9219]$。
- **对称量化（Q4_0）的局限**（以下步长按整个张量计算）：
  - 步长由绝对值最大的数决定：$d_{\text{sym}} = \frac{2.9219}{8} \approx \mathbf{0.365}$。
  - 16 个格点为 $-7d$ 至 $+8d$，即 $[-2.56, +2.92]$，**负半轴 7 个与 0 共 8 个在数据范围外**。
- **非对称量化（Q4_1）的修正**：
  - 记录独立最小值 $m = 0.0417$，步长按极差紧密划分：$d_{\text{asym}} = \frac{2.9219 - 0.0417}{15} \approx \mathbf{0.192}$。
  - **步长约为 Q4_0 的 1/1.9，16 个格点全部落在数据范围内**。
""")
    p.notes("""
刚才我们说对称量化硬件友好，但大家看屏幕上的真实数据：最终 RMSNorm 的 2048 个权重全是
正数，大部分在 2.2 到 2.6 之间。Q4_0 的格点以 0 为中心排布，16 个格点中有 8 个（负半轴 7 个
与 0）落在数据范围之外。实际的 GGUF 文件中，归一化层这类一维张量保留为 F32，不做量化；
这里用它演示数据全为正时对称格点的浪费。
""")


def symmetric_failure_grid(p):
    p.title("对称量化的不足：数据非对称分布下的精度雪崩")
    p.image("assets/symmetric-waste.svg", width_px=1100)
    p.notes(r"""
上面是最终 RMSNorm（`model.norm.weight`）2048 个权重的真实分布：全部为正，大部分集中在 2.2 到 2.6 之间。
下面两行把两种 4 位量化的 16 个格点画在同一条实数轴上，步长按整个张量计算。

Q4_0 的步长由最大绝对值决定，$d = 2.9219 / 8 \approx 0.365$，格点为 $-7d$ 到 $8d$。
负半轴的 7 个格点与 0 这个格点都在数据范围之外，数据范围内只剩 8 个格点。
Q4_1 以最小值 $m = 0.0417$ 为起点，把 $[0.0417, 2.9219]$ 等分为 15 段，
$d \approx 0.192$，16 个格点全部落在数据范围内，间距约为 Q4_0 的一半。
""")

def symmetric_failure_cont(p):
    p.title("对称量化的不足：数据非对称分布下的精度雪崩")
    p.slide("""
- `./quant_compare norm`，加载模型 `w-final-norm.bf16` 进行实测
""")
    p.demo("", """cd examples
./quant_compare norm""",
           output="""ext/w-final-norm.bf16   2048 weights   range +0.0417 .. +2.9219   rms 2.36650
  scheme        bytes  bits/w   rmse       rel.rmse  max err
  per-tensor     1026   4.01   0.094299     3.98%   0.181641
  per-256        1040   4.06   0.100978     4.27%   0.181641
  Q4_0           1152   4.50   0.100460     4.25%   0.175781
  Q4_1           1280   5.00   0.022288     0.94%   0.084106
  Q4_K           1152   4.50   0.051072     2.16%   0.094694""",
           bold=[5, 6],
           files=["examples/quant_compare.c"])
    p.slide("""
- **结论**：增加仅 0.5 位/权重的元数据存储最小值 $m$，相对误差从 **4.25% 显著降低至 0.94%**！
""").footnote("文献出处：Llama-3.2-1B Weight Distribution Analysis & GGUF Format Implementation.")


def clipping_tradeoff(p):
    p.title("离群点困境：动态范围与局部精度的零和博弈")
    p.slide(r"""
- **模型权重的钟形分布特征**：绝大部分数值密集聚集在 0 附近，两翼伴随少量极端离群值。
- **策略 A：MinMax 准则**
  - 强制覆盖最大极值：$w_{\max} = \max|W|$。
  - **弊端**：只要有一个离群尖刺，步长 $S = w_{\max}/q_{\max}$ 就会被暴力拉大，导致处于中间的绝大多数正常权重全被挤压在 0 附近的极少数离散桶中，**舍入误差剧增**。
- **策略 B：饱和截断准则**
  - 设定阈值 $T < \max|W|$，超出 $T$ 的数据强行截断为 $T$。
  - **收益**：步长 $S = T / q_{\max}$ 显著缩小，核心有效区域网格致密；
  - **代价**：两端尾部承担**截断误差**。
- **核心权衡**：选取 $T$，使截断误差与舍入误差之和最小。
""").footnote("文献出处：Zhao, R. et al. *Improving Neural Network Quantization without "
              "Retraining using Outlier Channel Splitting*, ICML (2019).")
    p.notes("""
截断阈值的选择是截断误差与舍入误差之间的权衡。MinMax 准则为了迁就极少数极端离群点，不惜
拉大整个量化网格刻度，占绝大多数的核心数据分辨率随之下降；饱和截断则把尾部极值截断到 $T$，
换取主体密集区间内更细的网格步长。$T$ 越小，舍入误差越小、截断误差越大，最优的 $T$ 使两者
之和最小。系统工程从不追求教条式的绝对无损，而是追求全分布期望误差的最小化。
""")



def granularity_spectrum(p):
    p.title("量化粒度：压缩率与精度的光谱权衡")
    p.slide(r"""
- **粒度谱系定义**：
  - **整张量粒度**：全矩阵数千万参数共用 1 个缩放系数。元数据极小，但若出现孤立异常值将导致全张量动态范围失真。
  - **按通道粒度**：矩阵每一行独立分配 1 个缩放系数（$M$ 行共 $M$ 个缩放系数）。
  - **分组量化**：将连续权重切分为若干固定大小小块（分组规模 $G$），各块独享参数。
- **分摊至单权重的元数据开销公式**：
  $$\text{元数据位数/权重} = \frac{\text{每组元数据位数}}{\text{分组规模 } G}$$
  - [例] FP16 缩放系数（16 位）分摊至 32 个权重：$16 \div 32 = \mathbf{0.5\text{ 位/权重}}$。
""").footnote("文献出处：Krishnamoorthi, R. *Quantizing deep convolutional networks for efficient "
              "inference: A whitepaper* (2018)；Dettmers, T. & Zettlemoyer, L. *The case for "
              "4-bit precision*, ICML (2023).")
    p.table(
        headers=["ffn_down 前 4096 个", "整段共用（4.00 位）", "256 元素分组（4.06 位）",
                 "32 元素分组（4.50 位）"],
        rows=[["**相对均方根误差**", "13.50%", "11.15%", "**8.42%**"]],
        align=["left", "center", "center", "center"],
    )
    p.notes("""
分组粒度越小，网格步长越贴合局部数据分布，拟合精度通常越高；但系统设计必须精确核算元数据
开销：每个分组都需独立保存一个 FP16 缩放系数。以 32 元素分组为例，分摊到每个参数上会额外
产生 0.5 位的存储成本。这部分开销即为换取局部表征精度所付出的体系结构代价。
表中数据是 Llama-3.2-1B 第 0 层 ffn_down 的前 4096 个权重，第一列是这 4096 个共用 1 个步长。
""")


def gguf_blocks(p):
    p.title("格式规范：GGUF 基础 4 位量化块物理排布")
    p.image("assets/gguf-q4-blocks.svg", width_px=880)
    p.slide("""
- **Q4_0 交叉配对排布规则**：
  - 单字节的低 4 位存放第 $j$ 个权重，高 4 位存放第 $j + 16$ 个权重（前后半块交织对齐）。
""")
    p.notes("""
这就是我们用 llama.cpp 运行时磁盘上权重的真实排布方式。Q4_0 结构非常规整，18 个字节
对应 32 个权重。请大家注意它打包时的精巧设计：第 0 个权重和第 16 个权重拼在同一个字节里，
高低半字节交错。这种交替排布非常契合后续向量化解包时的并行指令对齐。
""")


def gguf_blocks_cont(p):
    p.title("格式规范：GGUF 基础 4 位量化块物理排布")
    p.slide(r"""
- **GGUF Q4_0 块结构（对称量化基准）**：
  - 每 **32 个连续权重**划为 1 个量化块。
  - **`d` 字段**：2 字节（FP16 缩放步长）。
  - **`qs` 字段**：16 字节（32 个 4-bit 编码打包存储）。
  - **单块开销**：$2 + 16 = \mathbf{18\text{ 字节}}$ $\implies$ 平均每个权重占用 **4.50 位**。
- **GGUF Q4_1 块结构（非对称量化）**：
  - **`d` 字段**：2 字节（FP16 缩放步长）。
  - **`m` 字段**：2 字节（FP16 块内最小值）。
  - **`qs` 字段**：16 字节（32 个 4-bit 编码）。
  - **单块开销**：$2 + 2 + 16 = \mathbf{20\text{ 字节}}$ $\implies$ 平均每个权重占用 **5.00 位**。
""").footnote("文献出处：`llama.cpp` GGUF Specification & `ggml-quants.h`.")


def kquants_superblock(p):
    p.title("系统进阶：K-quants 超块设计与二级量化")
    p.slide(r"""
- **工程痛点**：
  - Q4_0 对称省空间但非对称数据下精度差；Q4_1 精度好但多占了 $11\%$ 的空间。
  - **能否用 Q4_0 的体积，实现接近 Q4_1 的非对称高精度？**
""")
    p.image("assets/q4-dilemma.svg",
            alt="位宽与误差的平面上，Q4_0 在 4.50 位、4.25%，Q4_1 在 5.00 位、0.94%；"
                "4.50 位、约 1% 处是一个带问号的虚线圆，两者都达不到",
            width_px=980)
    p.notes("""
图中两个点是「对称量化的不足」一节 `./quant_compare norm` 的实测结果。Q4_0 每 32 个权重占 18 字节，
相对误差 4.25%；Q4_1 多存一个 FP16 偏移，占 20 字节，误差降到 0.94%，体积多出
20 ÷ 18 − 1 ≈ 11%。虚线圆是两种格式都达不到的位置：Q4_0 的体积、Q4_1 的精度。
Q4_K 落在哪里，由本页第三面的表格给出。
""")


def kquants_superblock_cont(p):
    p.title("系统进阶：K-quants 超块设计与二级量化")
    p.image("assets/q4k-superblock.svg", width_px=940)
    p.slide(r"""
- **Q4_K 的破局之道：超块与二级量化**：
  - **超块规模**：每个超块包含 **256 个权重**，细分为 **8 个子块（各 32 个权重）**。
  - 8 个子块需要 8 个步长与 8 个偏移量（共 16 个参数）。
  - **二级量化机制**：
    - 步长以 `d` 为单位量化为 6 位整数 `sc`，偏移以 `dmin` 为单位量化为 6 位整数 `m`。
    - `sc` 与 `m` 共 $16 \times 6 = 96\text{ bits} = \mathbf{12\text{ 字节}}$；`d` 与 `dmin` 为 FP16，共 **4 字节**。
""")
    p.notes(r"""
请大家体会二级量化的系统工程巧思：量化对象不仅限于权重数据
本身，量化产生的缩放与偏移参数同样可以被再次量化。8 个子块原本需要 32 字节的浮点元数据，
经过 6 位二次量化后被紧凑压缩为 12 字节。在近似对称的权重上，Q4_K 以 4.5 位的存储开销，
取得了与 5.0 位的 Q4_1 相当的误差；全为正的权重上仍不及 Q4_1，原因见第三面的讲稿。

`d` 取 8 个子块步长中的最大值除以 63，`sc` = round(子块步长 ÷ `d`)，因此 `sc` 落在
0 ~ 63，正好 6 位；`dmin` 与 `m` 对偏移做同样的事。`d` 与 `dmin` 本身不再量化，按 FP16 存储。
""")


def kquants_superblock_cont2(p):
    p.title("系统进阶：K-quants 超块设计与二级量化")
    p.slide(r"""
- **还原一个权重**（$q$ 为 4 位编码，取值 0 ~ 15）：
  - 子块步长 = `d` × `sc`，子块偏移 = `dmin` × `m`；权重 = 子块步长 × $q$ − 子块偏移。
- **超块开销结算**：
  - 256 个 4 位权重编码（128 字节）+ 二级量化参数（12 字节）+ `d` 与 `dmin`（4 字节）= **144 字节**。
  - **平均位宽**：$144 \times 8 \div 256 = \mathbf{4.50\text{ 位/权重}}$（与 Q4_0 的存储开销完全一致）。
""").footnote("文献出处：Kawrakow, I. *k-quants*, llama.cpp PR #1684 (2023).")
    p.table(
        headers=["格式", "位/权重", "全正权重实测误差", "近似对称权重实测误差"],
        rows=[
            ["Q4_0", "4.50", "4.25%", "8.42%"],
            ["Q4_1", "5.00", "**0.94%**", "**7.69%**"],
            ["**Q4_K**", "**4.50**", "2.16%", "7.70%"],
        ],
        align=["left", "center", "center", "center"],
    )
    p.notes("""
表中数字是 `./quant_compare` 对 Llama-3.2-1B 两段真实权重的输出，误差为相对均方根误差。`w-down-proj` 上，
Q4_K 以 4.50 位取得 7.70%，与 5.00 位的 Q4_1（7.69%）几乎相同。`w-final-norm` 上，
Q4_K 把 Q4_0 的 4.25% 降到 2.16%，仍高于 Q4_1 的 0.94%：Q4_K 的偏移只能把格点向下平移
（子块最小值大于 0 时按 0 处理，与 llama.cpp 一致），全为正的数据用不上偏移，16 个格点
铺在 0 到最大值之间；Q4_1 的格点从数据的最小值开始，因此更密。
""")


def scale_bits(p):
    p.title("位级别的代价：6 位字段跨越字节边界的重构")
    p.image("assets/q4k-scale-bits.svg", width_px=960)
    p.slide("""
- **严峻的物理现实**：
  - 16 个 6 位参数共 96 bits，刚好凑成整整 12 个字节。但 **6 与 8 无法整除**！
  - 这意味着：**字段必须跨越物理字节边界存储，不能有任何一位被浪费。**
""")
    p.notes("""
天下没有免费的午餐。要省下这几个字节，代价就是代码变复杂。大家看 `sc4` 的提取：低 4 位在
第 8 字节，高 2 位在第 0 字节的高处！如果你的位移写错一位，权重直接全偏。没有扎实的掩码和
移位基本功，你连工业级大模型的文件都读不懂。
""")


def scale_bits_cont(p):
    p.title("位级别的代价：6 位字段跨越字节边界的重构")
    p.slide(r"""
- **GGUF `scales` 12 字节物理布局协议**：
  - 前 8 个字节（`scales[0..7]`）：低 6 位分别存放前 4 个步长 $sc_{0..3}$ 与前 4 个偏移 $m_{0..3}$。
  - 空出的高 2 位（共 $8 \times 2 = 16$ 位）：正好用来存放后 4 个步长与偏移的**高 2 位**！
  - 后 4 个字节（`scales[8..11]`）：低 4 位与高 4 位分别存放后 4 个步长与偏移的**低 4 位**！
- **C 语言解码内核实现**：
""").footnote("文献出处：`llama.cpp/k_quants.c` Decode Kernel.")
    p.code("c", K_SCALES_DECODE)



def ai_float_formats(p):
    p.title("浮点重构：从通用规范到专用 AI 浮点")
    p.slide(r"""
- **位域演变对比**：
  - **FP32**：`[1s]` + `[8 exp]` + `[23 mantissa]`
  - **FP16**：`[1s]` + `[5 exp]` + `[10 mantissa]` （动态范围较小，易溢出）
  - **BF16**：`[1s]` + `[8 exp]` + `[ 7 mantissa]` （保全 8 位阶码，动态范围与 FP32 相同）
- **NVIDIA Hopper 时代的 1 字节浮点：FP8 双子星规范**：
  1. **E4M3（1 位符号，4 位阶码，3 位尾数）**：
     - 尾数多 1 位，精度较高，**被建议用于前向传播的权重与激活值**。
     - **突破 IEEE 规范**：废除无穷大（$\pm\infty$）编码，将保留码段释放用来表示更大正常数（最大数达 448）。
  2. **E5M2（1 位符号，5 位阶码，2 位尾数）**：
     - 阶码与 FP16 一致，动态范围较大；**被建议用于反向传播的梯度**。
""").footnote("文献出处：Micikevicius, P. et al. *FP8 Formats for Deep Learning*, "
              "arXiv:2209.05433 (2022).")
    p.notes("""
体系结构的发展展现了对传统规范的务实重塑。BF16 缩减尾数，保留与 FP32 相同的 8 位阶码，
动态范围与 FP32 相同；而到了 FP8 时代，E4M3 甚至废除了对无穷大的编码保留，将其重新分配给有限
数值以扩展表示范围。E4M3 与 E5M2 的分工是 FP8 论文的建议，实际的训练与推理方案不止这一种。
这表明在特定领域计算中，数据表示形式始终服务于具体的系统效能与动态范围需求。
""")


def nf4_lut(p):
    p.title("非均匀量化：NF4 的信息论等熵分割")
    p.slide(r"""
- **均匀整数量化的局限**：
  - 预训练权重按块归一化后，**近似服从零均值正态分布**（QLoRA 的前提）。
  - 等距网格在两端低概率区与中间高概率区分配同样多的格点，中间区域分辨率相对不足。
- **分位数量化原则**：
  - 让每个编码被取到的概率相等，4 位编码的信息熵达到最大。
- **NF4 量化实现机制**：
  - 取标准正态分布的分位数作为 16 个格点，缩放到 $[-1, 1]$；为使 0 能被精确表示，负半轴取 7 个、正半轴取 8 个，各区间只是近似等概率。
  - **查找表还原**：
    - 4 位存储的不再是数值，而是 **0 ~ 15 的查找表索引**；
    - 运行时查一张 16 个常量的表，乘以块的缩放系数（absmax）后送入计算。
""").footnote("文献出处：Dettmers, T. et al. *QLoRA: Efficient Finetuning of Quantized "
              "LLMs*, NeurIPS (2023).")
    p.notes("""
NF4 不要求硬件支持非等距运算：内存里存 4 位索引，计算前查一张 16 个值的表，再乘以该块的
缩放系数（块内绝对值的最大值），还原为浮点。
16 个值取自标准正态分布的分位数，前提是按块归一化后的权重近似服从零均值正态分布；为了让
0 能被精确表示，负半轴 7 个、正半轴 8 个，因此各区间只是近似等概率。权重分布偏离正态时，
这组格点不再是最优的选择。
""")


def lab_release(p):
    p.title("小练习：动手打造你自己的量化引擎")
    p.slide("""
- **实验对象**：`Qwen/Qwen3-VL-2B-Instruct` 的 safetensors 文件，625 个 BF16 张量，其中 310 个属于语言模型。
- **四个部分**：
  - **A 读字节（15 分）**：用移位组合小端整数读出头部长度，不得用 `memcpy`；实现 BF16、FP16 到 FP32 的展开与 FP32 到 FP16 的舍入。
  - **B 三种量化格式（35 分）**：Q4_0、Q4_1、Q4_K 的量化与反量化，输出块与期望值逐字节相同；实现 6 位 `sc`、`m` 的打包。
  - **C 全模型量化与字节统计（30 分）**：跳过 315 个视觉张量；`plan` 预测的总字节数与产物大小完全相等，产物 md5 与期望值相同。
  - **D 加载运行（15 分）**：产物组装为 GGUF，由 ollama 以学号为输入贪心生成 128 个 token，提交生成文本的 md5。
""").footnote("文献出处：ICS 习题课一实验题面 nano-quant；Qwen/Qwen3-VL-2B-Instruct 模型卡（Apache-2.0）。")
    p.notes("""
本实验把第七、第八部分讲过的三种格式用在一个真实模型上。学生不需要写 Python，也不需要自己
实现推理：读取 safetensors 的小端头部，完成三种 4 位格式的编解码与全模型量化，最后把产物交给
ollama 运行，用生成文本的 md5 自查。A 部分与 Q4_0 在习题课上跟着本讲的幻灯片完成，其余部分
课后完成，周期两周。
""")
