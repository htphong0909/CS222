# KẾ HOẠCH BÀI GIẢNG SLIDE 60 PHÚT (45 SLIDES): NỀN TẢNG LÝ THUYẾT, SƠ ĐỒ LUỒNG DỮ LIỆU & ĐỘNG CƠ TIẾN HÓA POST-TRAINING CHO LLMs

**Thời lượng:** 60 phút (52 phút trình bày 45 slides $\approx$ 1.15 phút/slide + 8 phút Thảo luận Q&A)  
**Trọng tâm thiết kế:**
1. **Làm rõ Động cơ Tiến hóa (Limitations $\to$ Motivation):** Mọi bước chuyển giữa các phương pháp đều xuất phát từ **giới hạn lý thuyết của phương pháp cũ** để dẫn dắt tự nhiên sang **động cơ ra đời phương pháp mới**.
2. **Sơ đồ Luồng Dữ liệu & Kiến trúc Trực quan (Dataflow Diagrams):** Có các slide chuyên biệt vẽ sơ đồ khối và luồng đi của dữ liệu/đạo hàm cho **SFT, Reward Model, RLHF + PPO (4 mô hình), DPO (2 mô hình), RLVR + GRPO (Nhóm $G$ mẫu), và Distillation (3 cấp độ)**.
3. **Toán học Cốt lõi & Dẫn xuất Chi tiết từ RLHF $\to$ DPO:** Chỉ giữ các công thức cốt lõi nhất, đặc biệt bổ sung chuỗi 3 slides giải thích từng bước biến đổi đại số từ công thức **RLHF có phạt KL $\longrightarrow$ Nghiệm đóng $\pi^*$ $\longrightarrow$ Triệt tiêu $Z(x)$ $\longrightarrow$ Công thức DPO**.

---

## BẢNG TỔNG QUAN CẤU TRÚC 45 SLIDES & CÁC BƯỚC CHUYỂN GIAO PHƯƠNG PHÁP

| Phần | Số Slide | Thời lượng | Chuỗi Tiến hóa Phương pháp (Giới hạn Cũ $\to$ Động cơ Mới) | Sơ đồ & Toán học Trọng tâm |
| :--- | :---: | :---: | :--- | :--- |
| **Phần 1: Từ Pre-training đến SFT** | **Slide 1 – 8** *(8 slides)* | *09 phút* | **Pre-training** *(Lệch mục tiêu: chỉ hoàn thành văn bản)* $\longrightarrow$ **SFT** *(Học bắt chước có điều kiện)* $\longrightarrow$ **3 Giới hạn của SFT** *(Exposure Bias, Ảo giác, Thiếu phản hồi âm)*. | • Sơ đồ Luồng dữ liệu SFT & Mặt nạ $m_t$<br>• Công thức (1) Pre-training & (2) SFT |
| **Phần 2 (Nhánh A): Tối ưu hóa Sở thích — Từ RLHF đến DPO** | **Slide 9 – 24** *(16 slides)* | *20 phút* | **SFT** $\longrightarrow$ **Reward Modeling** *(Chấm điểm so sánh cặp)* $\longrightarrow$ **RLHF phạt KL** *(Chống Reward Hacking)* $\longrightarrow$ **DPO** *(Giải nghiệm đóng toán học, bỏ Reward Model & bỏ vòng lặp RL)* $\longrightarrow$ **SimPO / RPO** *(Trị bệnh nói dài & tụt xác suất)*. | • Sơ đồ Huấn luyện Reward Model<br>• Sơ đồ 4 mô hình RLHF + PPO<br>• **Chuỗi 3 Slides dẫn xuất Toán từ RLHF $\to$ DPO**<br>• Sơ đồ 2 mô hình của DPO |
| **Phần 3 (Nhánh B): Học Tăng cường & Suy luận — PPO, RLVR & GRPO** | **Slide 25 – 37** *(13 slides)* | *15 phút* | **Offline DPO** *(Không tự khám phá được)* $\longrightarrow$ **Online RL (Policy Gradient & PPO)** $\longrightarrow$ **RLVR** *(Thay giám khảo mạng nơ-ron bằng trình kiểm chứng khách quan)* $\longrightarrow$ **ORM + `<think>`** *(Giải phóng suy luận, bỏ PRM)* $\longrightarrow$ **GRPO** *(Thay mạng Critic bằng Trung bình Nhóm $G$ câu)*. | • Sơ đồ Chu trình RL & PPO-Clip<br>• Sơ đồ Kiến trúc RLVR + GRPO ($G$ nhánh)<br>• Công thức Policy Gradient, PPO-Clip & GRPO |
| **Phần 4 (Nhánh C): Chưng cất Mô hình & Tổng kết** | **Slide 38 – 45** *(8 slides)* | *08 phút* | **Tự chạy RL trên Mô hình Nhỏ** *(Không mò ra lời giải đúng)* $\longrightarrow$ **Model Distillation** *(Kế thừa từ mô hình Thầy)* $\longrightarrow$ **Reverse KL** *(Chống ôm đồm quá tải của Forward KL)*. | • Sơ đồ 3 Hình thức Chưng cất (Logits, Outputs, Reasoning)<br>• Công thức Softmax nhiệt độ $\tau$ & Forward/Reverse KL |

---

## NỘI DUNG CHI TIẾT TỪNG SLIDE (SLIDE 1 – SLIDE 45)

---

### PHẦN 1: TỪ PRE-TRAINING ĐẾN SFT & CÁC GIỚI HẠN CỦA HỌC BẮT CHƯỚC *(Slides 1 – 8)*

#### **Slide 1: Trang bìa & Câu hỏi Lớn của Post-Training**
* **Nội dung trên Slide:**
  * **Tiêu đề:** Nền tảng Lý thuyết Post-Training cho Mô hình Ngôn ngữ Lớn (LLMs).
  * **Ba trụ cột lý thuyết:** (A) Tối ưu hóa Sở thích & Căn chỉnh $\cdot$ (B) Học Tăng cường & Suy luận $\cdot$ (C) Chưng cất Mô hình.
  * **Câu hỏi xuyên suốt:** *Tại sao các phương pháp Post-Training lại liên tục tiến hóa từ SFT $\to$ RLHF/PPO $\to$ DPO $\to$ RLVR/GRPO $\to$ Distillation?*
* **Lời giảng giải:**
  * Thay vì học thuộc lòng từng thuật toán rời rạc, bài giảng hôm nay sẽ đi theo **sợi dây tiến hóa lý thuyết**: ở mỗi giai đoạn, chúng ta sẽ chỉ ra **giới hạn chí mạng của phương pháp cũ** là gì, và giới hạn đó đã thúc đẩy các nhà khoa học phát minh ra **phương pháp mới** như thế nào.

---

#### **Slide 2: Điểm Xuất phát — Bản chất và Giới hạn của Pre-training (*Objective Mismatch*)**
* **Nội dung trên Slide:**
  * **Công thức Cốt lõi số 1 (Hàm mất mát Pre-training — Dự đoán từ tiếp theo):**
    $$\mathcal{L}_{\text{Pretrain}}(\theta) = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, x_2, \dots, x_{t-1})$$
  * **[GIỚI HẠN CỦA PRE-TRAINING $\longrightarrow$ ĐỘNG CƠ CỦA POST-TRAINING]:**
    * *Giới hạn:* Mô hình nền (Base Model) chỉ là một **Bộ hoàn thành văn bản theo xác suất thống kê** trên Internet. Khi người dùng nhập một câu hỏi, nó có xu hướng sinh tiếp các câu hỏi khác giống như một đề thi trên mạng chứ không trả lời yêu cầu.
    * *Động cơ:* Cần một giai đoạn huấn luyện hậu kỳ (**Post-training**) để chuyển mục tiêu từ *"viết tiếp văn bản ngẫu nhiên"* sang *"phản hồi có điều kiện thỏa mãn yêu cầu con người"*.

---

#### **Slide 3: Bản đồ Cây Phân loại Lý thuyết Post-Training**
* **Nội dung trên Slide:**
  ```text
  Dữ liệu Web ──► Pre-training (Base Model)
                       │
                       ▼  [Khởi tạo định dạng hội thoại bằng SFT]
              ┌────────┴────────┐
              ▼                 ▼                 ▼
   [A. PREFERENCE &     [B. RL & REASONING     [C. MODEL
    ALIGNMENT]           ENHANCEMENT]           DISTILLATION]
   • Reward Modeling    • RLHF vs. RLVR        • Logit Distillation
   • RLHF               • ORM vs. PRM          • Sequence Distillation
   • DPO & Biến thể     • Policy Gradient      • Reasoning / Rationale
                        • PPO vs. GRPO           Distillation
  ```
* **Lời giảng giải:**
  * Đây là bản đồ toàn cảnh của buổi thuyết trình: Khởi đầu từ SFT để định hình giao thức hỏi–đáp, sau đó đi sâu vào 3 trụ cột lớn (A: Dạy mô hình biết câu trả lời nào là tốt; B: Dạy mô hình tự suy luận qua phần thưởng; C: Truyền thụ năng lực từ mô hình lớn sang mô hình nhỏ).

---

#### **Slide 4: Phương pháp Đầu tiên — Supervised Fine-Tuning (SFT) & Sơ đồ Luồng Dữ liệu**
* **Nội dung trên Slide:**
  * **Bản chất Lý thuyết:** Học bắt chước hành vi chuyên gia (*Behavioral Cloning*).
  * **Sơ đồ Luồng Dữ liệu của SFT:**
    ```text
    [Dữ liệu Chuyên gia viết sẵn]
      Câu hỏi x: "<|user|> Thủ đô của Pháp là gì? <|assistant|>"   ──► Mặt nạ m_t = 0 (Bỏ qua, không tính lỗi)
      Đáp án y*: "Thủ đô của Pháp là Paris. <|end|>"               ──► Mặt nạ m_t = 1 (Tính lỗi Cross-Entropy)
                                                                              │
                                                                              ▼
                                                        [Mô hình LLM π_θ] ◄── Cập nhật Gradient (Kéo xác suất y* lên 100%)
    ```
* **Lời giảng giải:**
  * Nhìn vào sơ đồ luồng dữ liệu: Toàn bộ câu hỏi $x$ và câu trả lời mẫu $y^*$ được ghép thành một chuỗi duy nhất có gắn các thẻ đánh dấu vai trò (`<|user|>`, `<|assistant|>`, `<|end|>`). Tuy nhiên, khi tính hàm mất mát để cập nhật trọng số, một mặt nạ nhị phân $m_t$ sẽ khóa toàn bộ phần câu hỏi lại và chỉ cho phép đạo hàm truyền ngược từ phần câu trả lời $y^*$.

---

#### **Slide 5: Công thức Toán học của SFT — Hợp lý Cực đại Có điều kiện (*Conditional MLE*)**
* **Nội dung trên Slide:**
  * **Công thức Cốt lõi số 2 (Hàm mất mát SFT có Mặt nạ $m_t \in \{0, 1\}$):**
    $$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} m_t \cdot \log \pi_\theta(x_t \mid x_{<t}) = -\sum_{t \in \text{Response } y^*} \log \pi_\theta(y_t^* \mid x, y_{<t}^*)$$
  * **Tại sao chỉ tính hàm mất mát trên phần trả lời $y^*$?**
    * Vì xác suất chung $\log P(x, y^*) = \log P(x) + \log \pi_\theta(y^* \mid x)$. Câu hỏi $x$ là điều kiện cho trước bởi người dùng; mô hình vẫn đọc $x$ ở phía sau dấu gạch đứng $(\mid x, y_{<t}^*)$ để hiểu ngữ cảnh, nhưng chỉ cần tối ưu hóa xác suất sinh ra câu trả lời $y^*$.

---

#### **Slide 6: Bản chất Tri thức trong SFT — Giả thuyết Căn chỉnh Bề mặt (*LIMA*)**
* **Nội dung trên Slide:**
  * **Giả thuyết Căn chỉnh Bề mặt (*Superficial Alignment Hypothesis — Zhou et al., 2023*):**
    * Tri thức sự kiện và năng lực ngôn ngữ đã nằm sẵn trong mô hình từ giai đoạn Pre-training.
    * SFT chủ yếu dạy mô hình **"khuôn mẫu giao tiếp bề mặt"** (cách mở bài, cách trình bày, khi nào dừng lại ở thẻ `<|end|>`).
  * **Hệ quả lý thuyết:** Trong SFT, **chất lượng mẫu quan trọng hơn số lượng mẫu**. Khi dùng dữ liệu tổng hợp do AI sinh ra (*Synthetic Data*), bắt buộc phải đi kèm bộ lọc tuyển chọn (**Rejection Sampling**) để tránh hiện tượng thoái hóa phân phối (**Model Collapse**).

---

#### **Slide 7: [CHUYỂN GIAO 1] Giới hạn số 1 của SFT — Hiện tượng Lệch Phân phối (*Exposure Bias*)**
* **Nội dung trên Slide:**
  * **Sơ đồ Sự Vênh nhau giữa Lúc Học và Lúc Chạy Thực tế của SFT:**
    ```text
    LÚC HỌC (Teacher Forcing):
      Từ mẫu chuẩn y*_1 ──► Mô hình đoán y_2  |  Từ mẫu chuẩn y*_2 ──► Mô hình đoán y_3
      (Luôn được nhìn tiền tố đúng 100% của chuyên gia)

    LÚC CHẠY THỰC TẾ (Autoregressive Generation):
      Mô hình tự sinh ŷ_1 ──► Mô hình đoán ŷ_2 (dựa trên ŷ_1) ──► Mô hình đoán ŷ_3 (dựa trên ŷ_1, ŷ_2)
      (Nếu ŷ_1 hơi chệch ──► Ngữ cảnh lạ chưa từng học ──► Lỗi tích lũy dây chuyền O(T²))
    ```
* **Lời giảng giải:**
  * Tại sao một mô hình học SFT điểm rất cao nhưng khi chạy thực tế viết câu dài lại hay bị lạc đề? Đó là vì lúc học (*Teacher Forcing*), ở mỗi bước chữ, mô hình luôn được "đút" cho các chữ phía trước chuẩn 100% của chuyên gia. Nó chưa bao giờ được dạy cách tự xử lý khi chính nó lỡ viết chệch một chữ!

---

#### **Slide 8: [CHUYỂN GIAO 1 - Tiếp] Giới hạn số 2 & 3 của SFT $\longrightarrow$ Động cơ ra đời Nhánh A & B**
* **Nội dung trên Slide:**
  * **Giới hạn 2 — Bắt chước mù quáng gây Ảo giác (*Hallucination*):**
    * Ép mô hình SFT học thuộc lòng các đáp án chứa thông tin nằm ngoài tri thức Pre-training của nó sẽ dạy mô hình thói quen: *"Dù không biết câu trả lời cũng phải bịa ra một đáp án nghe thật trôi chảy!"*
  * **Giới hạn 3 — Thiếu Tín hiệu Phản hồi Âm (*No Negative Signal*):**
    * SFT chỉ kéo xác suất của 1 câu mẫu $y^*$ lên, còn mọi câu trả lời khác (dù là diễn đạt đồng nghĩa rất hay, hay là câu sai sự thật nghiêm trọng) đều bị đối xử mù quáng như nhau.
  * **$\Longrightarrow$ ĐỘNG CƠ CHUYỂN SANG POST-TRAINING NÂNG CAO:**
    1. Cần một cơ chế **so sánh và phân biệt rõ câu Tốt ($+$) với câu Xấu ($-$)** $\longrightarrow$ Sinh ra **Nhánh A (Preference & Alignment)**.
    2. Cần cho mô hình **tự mình viết thử câu trả lời rồi mới chấm điểm** để xóa bỏ *Exposure Bias* $\longrightarrow$ Sinh ra **Nhánh B (Reinforcement Learning)**.

---

### PHẦN 2 (NHÁNH A): TỐI ƯU HÓA SỞ THÍCH & CĂN CHỈNH — TỪ RLHF ĐẾN DPO *(Slides 9 – 24)*

#### **Slide 9: Tổng quan Nhánh A (*Preference & Alignment Optimization*)**
* **Nội dung trên Slide:**
  * **Câu hỏi trung tâm:** *"Làm sao định nghĩa và dạy mô hình hướng tới những phản hồi đáng mong muốn khi không có một đáp án mẫu duy nhất?"*
  * **Hai bước tư duy của Nhánh A:**
    * **A1. Học Sở thích (*Learning Preferences*):** Thu thập các phán quyết so sánh cặp $(y_w \succ y_l)$ từ Con người hoặc AI $\to$ Xây dựng **Mô hình Phần thưởng (*Reward Model*)**.
    * **A2. Tối ưu hóa theo Sở thích (*Optimizing for Preferences*):** Dùng **RLHF** (Gián tiếp qua Reward Model + RL) hoặc **DPO** (Trực tiếp từ cặp sở thích).

---

#### **Slide 10: A1. Tại sao Thu thập Sở thích bằng So sánh Cặp (*Pairwise Comparison*)?**
* **Nội dung trên Slide:**
  * **Hạn chế của việc bắt con người chấm điểm tuyệt đối (1–10):**
    * Không có thước đo chuẩn chung giữa những người chấm khác nhau; cùng một người chấm buổi sáng và buổi chiều cũng cho điểm khác nhau.
  * **Động cơ dùng So sánh Cặp ($y_w \succ y_l$):**
    * Đặt 2 câu trả lời cho cùng câu hỏi $x$ cạnh nhau và chỉ hỏi: **"Câu nào tốt hơn?"**
    * Tạo ra bộ dữ liệu sở thích chuẩn: $(x, \, y_w \text{ [Chosen/Thắng]}, \, y_l \text{ [Rejected/Thua]})$.
  * **Hai nguồn gán nhãn:** Con người (**RLHF** — chính xác nhưng đắt) vs. AI chấm theo bộ quy tắc (**RLAIF / Constitutional AI** — mở rộng quy mô lớn).

---

#### **Slide 11: Sơ đồ Kiến trúc & Luồng Dữ liệu Huấn luyện Reward Model ($r_\phi$)**
* **Nội dung trên Slide:**
  * **Sơ đồ Luồng Dữ liệu Huấn luyện Mô hình Phần thưởng (Reward Model):**
    ```text
                        ┌──► [ câu thắng (x, y_w) ] ──► [ Reward Model r_φ ] ──► Điểm số thực r_φ(x, y_w) (VD: +3.5) ──┐
    Cùng câu hỏi x ─────┤                                 (Tháo LM Head,                                               ├──► Hiệu số: r_w - r_l
                        └──► [ câu thua  (x, y_l) ] ──►   gắn Value Head 1 chiều                                       │         │
                                                          tại token <EOS>)   ──► Điểm số thực r_φ(x, y_l) (VD: -1.2) ──┘         ▼
                                                                  ▲                                                    Hàm mất mát: -log σ(r_w - r_l)
                                                                  └──────────── Cập nhật Gradient φ (Ép r_w > r_l) ──────────────┘
    ```
* **Lời giảng giải:**
  * Sơ đồ này cho thấy cách một Reward Model hoạt động: Bản thân $r_\phi$ có khung xương là một mô hình ngôn ngữ, nhưng thay vì dự đoán từ tiếp theo, ở vị trí token cuối cùng `<EOS>` (nơi đã đọc hết cả câu hỏi và câu trả lời), nó nén thông tin thành **1 con số điểm duy nhất**. Trong lúc học, ta đưa cả câu thắng $y_w$ và câu thua $y_l$ đi qua $r_\phi$, rồi cập nhật trọng số $\phi$ sao cho điểm của $y_w$ phải vượt xa điểm của $y_l$.

---

#### **Slide 12: Toán học của Reward Modeling — Mô hình Xác suất Bradley-Terry**
* **Nội dung trên Slide:**
  * **Công thức Cốt lõi số 3 (Mô hình Bradley-Terry & Hàm mất mát Reward Model):**
    $$P(y_w \succ y_l \mid x) = \frac{\exp\big(r_\phi(x, y_w)\big)}{\exp\big(r_\phi(x, y_w)\big) + \exp\big(r_\phi(x, y_l)\big)} = \sigma\Big( r_\phi(x, y_w) - r_\phi(x, y_l) \Big)$$
    $$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \Big[ \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \Big]$$
  * **Tính chất Bất biến Tịnh tiến (Cực kỳ quan trọng cho DPO):**
    * Nếu cộng thêm một đại lượng bất kỳ $C(x)$ chỉ phụ thuộc vào câu hỏi $x$ vào điểm thưởng của mọi câu trả lời, hiệu số $\big[r_\phi(x, y_w) + C(x)\big] - \big[r_\phi(x, y_l) + C(x)\big] = r_\phi(x, y_w) - r_\phi(x, y_l)$ **không hề thay đổi**!

---

#### **Slide 13: [CHUYỂN GIAO 2] Từ Reward Model sang RLHF — Tại sao Tối ưu Điểm thưởng Ngây thơ lại Thất bại?**
* **Nội dung trên Slide:**
  * Khi đã có giám khảo $r_\phi(x, y)$, ý tưởng tự nhiên nhất là bảo mô hình ngôn ngữ $\pi_\theta$: *"Hãy tự sinh câu trả lời $y$ sao cho điểm $r_\phi(x, y)$ cao nhất!"*
  * **[GIỚI HẠN CỦA TỐI ƯU HÓA ĐIỂM THƯỞNG THUẦN TÚY — *REWARD HACKING*]:**
    * Theo **Định luật Goodhart**, vì $r_\phi(x, y)$ chỉ là một mạng nơ-ron không hoàn hảo nên nó luôn có những "góc khuất" chấm điểm sai.
    * Nếu không có rào chắn, mô hình $\pi_\theta$ sẽ nhanh chóng biến chất: nó phát hiện ra chỉ cần viết câu trả lời dài lê thê, nịnh hót, hoặc lặp đi lặp lại các từ khóa kỳ dị là lừa được $r_\phi$ cho điểm tối đa, trong khi câu văn thực tế trở thành rác (**Reward Overoptimization / Reward Hacking**)!
  * **$\Longrightarrow$ ĐỘNG CƠ:** Bắt buộc phải gắn thêm một **"sợi dây xích an toàn"** giữ cho mô hình mới $\pi_\theta$ không được phép đi quá xa khỏi mô hình SFT gốc $\pi_{\text{ref}}$!

---

#### **Slide 14: Công thức Toán học của RLHF — Ràng buộc Phân kỳ KL (*KL-Constrained RL*)**
* **Nội dung trên Slide:**
  * **Công thức Cốt lõi số 4 (Hàm mục tiêu RLHF có Điều chuẩn Phân kỳ KL):**
    $$\max_{\pi_\theta} \; \mathcal{J}_{\text{RLHF}}(\pi_\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \left[ \underbrace{r_\phi(x, y)}_{\text{Điểm thưởng từ Giám khảo}} - \beta \underbrace{\log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}}_{\text{Khoản phạt KL khi đi lệch khỏi bản gốc}} \right]$$
  * **Giải thích ý nghĩa từng thành phần:**
    * $r_\phi(x, y)$: Khuyến khích mô hình sinh ra câu trả lời có điểm chất lượng cao.
    * $\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$: Khoản phạt tỷ lệ với khoảng cách Kullback-Leibler ($D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$). Nếu mô hình mới $\pi_\theta$ sinh ra những câu văn kỳ quặc mà mô hình gốc $\pi_{\text{ref}}$ gán xác suất rất nhỏ ($\pi_{\text{ref}} \approx 0$), tỷ số $\frac{\pi_\theta}{\pi_{\text{ref}}} \to \infty$ sẽ tạo ra một hình phạt cực nặng!

---

#### **Slide 15: Sơ đồ Kiến trúc & Luồng Dữ liệu của Hệ thống RLHF + PPO Cổ điển (4 Mô hình)**
* **Nội dung trên Slide:**
  * **Sơ đồ Luồng Hoạt động của RLHF (với thuật toán PPO):**
    ```text
                                      ┌──► [ 2. Reference Model π_ref ] (Đóng băng) ──► Tính log π_ref(y|x) ──┐
                                      │                                                                       ▼
    Prompt x ──► [ 1. Actor / Policy π_θ ] ──► Tự sinh câu trả lời y ──────────────────────────────► [ Phạt KL: -β log(π_θ / π_ref) ]
                   (Đang huấn luyện)  │                                                                       │
                          ▲           ├──► [ 3. Reward Model r_φ ]      (Đóng băng) ──► Điểm thưởng r_φ(x,y) ─┼──► Tổng thưởng R(x,y)
                          │           │                                                                       ▼
                          │           └──► [ 4. Critic / Value V_ψ ]    (Đang huấn luyện) ──► Dự đoán điểm ──► Tính Lợi thế A_t
                          │                                                                                         │
                          └──────────────────────── Cập nhật Gradient PPO-Clip ◄────────────────────────────────────┘
    ```
* **Lời giảng giải:**
  * Hãy nhìn vào sơ đồ của RLHF + PPO: Để thực hiện một vòng lặp học duy nhất, hệ thống phải duy trì cùng lúc **4 mô hình khổng lồ**: (1) Mô hình chính *Actor* $\pi_\theta$ tự gõ từng chữ để sinh câu trả lời $y$, (2) Mô hình gốc *Reference* $\pi_{\text{ref}}$ tính xác suất gốc để phạt KL, (3) Mô hình giám khảo *Reward* $r_\phi$ chấm điểm cuối câu, và (4) Mô hình *Critic* $V_\psi$ ước lượng điểm kỳ vọng để tính lợi thế cập nhật.

---

#### **Slide 16: [CHUYỂN GIAO 3] Giới hạn của Hệ thống RLHF + PPO $\longrightarrow$ Động cơ ra đời DPO**
* **Nội dung trên Slide:**
  * **[3 GIỚI HẠN LỚN CỦA QUY TRÌNH RLHF + PPO]:**
    1. **Quy trình 2 giai đoạn cồng kềnh & Sai số chồng chất:** Phải huấn luyện Reward Model $r_\phi$ trước; nếu $r_\phi$ học sai ở đâu thì giai đoạn RL phía sau sẽ khuếch đại lỗi sai đó lên.
    2. **Gánh nặng 4 mô hình đồng thời:** Việc phối hợp 2 mô hình đang học ($\pi_\theta, V_\psi$) và 2 mô hình đóng băng ($\pi_{\text{ref}}, r_\phi$) cực kỳ tốn kém tài nguyên.
    3. **Sự bất ổn định của vòng lặp Online RL:** Phải chờ mô hình tự sinh văn bản từng từ một rất chậm chạp, và đạo hàm RL có độ nhiễu (phương sai) rất cao, dễ sập nếu chỉnh sai siêu tham số.
  * **$\Longrightarrow$ ĐỘNG CƠ RA ĐỜI DPO (Rafailov et al., 2023):**
    * *"Liệu ta có thể giải trực tiếp bài toán toán học của RLHF (Công thức 4) trên giấy để tìm ra mối liên hệ trực tiếp giữa mô hình ngôn ngữ $\pi_\theta$ và dữ liệu cặp $(y_w, y_l)$, qua đó **bỏ luôn cả bước huấn luyện Reward Model lẫn vòng lặp RL** không?"*

---

#### **Slide 17: Toán học từ RLHF $\to$ DPO (Bước 1/3): Nghiệm Đóng Tối ưu $\pi^*(y \mid x)$ của Bài toán RLHF**
* **Nội dung trên Slide:**
  * Xét lại bài toán cực đại hóa hàm mục tiêu RLHF có phạt KL (Công thức 4):
    $$\max_{\pi} \; \mathbb{E}_{y \sim \pi(\cdot \mid x)} \left[ r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]$$
  * **Chia cả biểu thức cho $\beta > 0$ và biến đổi logarit:**
    $$\max_{\pi} \; \mathbb{E}_{y \sim \pi(\cdot \mid x)} \left[ \log \exp\left(\frac{1}{\beta}r(x, y)\right) - \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right] = \min_{\pi} \; D_{\text{KL}}\left( \pi(y \mid x) \;\Big\|\; \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right) \right)$$
  * Vì khoảng cách $D_{\text{KL}}$ nhỏ nhất (bằng $0$) khi hai phân phối bằng nhau hoàn toàn, **nghiệm tối ưu chính xác $\pi^*(y \mid x)$** có công thức:
    $$\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right), \qquad \text{với } Z(x) = \sum_{y'} \pi_{\text{ref}}(y' \mid x) \exp\left(\frac{1}{\beta} r(x, y')\right)$$
* **Lời giảng giải:**
  * Đây là Bước 1 của phép biến đổi từ RLHF sang DPO: Hóa ra không cần chạy thuật toán PPO, ta vẫn biết chính xác hình dáng toán học của mô hình ngôn ngữ tối ưu $\pi^*(y \mid x)$!
  * Công thức nghiệm đóng này cực kỳ dễ hiểu: xác suất tối ưu $\pi^*(y \mid x)$ chính là xác suất của mô hình gốc $\pi_{\text{ref}}(y \mid x)$ được **nhân bội lên với hàm mũ của điểm thưởng** $\exp(r(x,y)/\beta)$, rồi chia cho mẫu số tổng $Z(x)$ (gọi là *Hàm phân hoạch — Partition Function*) để tổng xác suất các câu trả lời bằng $1$.

---

#### **Slide 18: Toán học từ RLHF $\to$ DPO (Bước 2/3): Đổi biến Rút $r(x, y)$ & Sự Triệt tiêu Kỳ diệu của $Z(x)$**
* **Nội dung trên Slide:**
  * **Tại sao trước đây không ai tính trực tiếp $\pi^*(y \mid x)$ ở Slide 17?**
    * Vì mẫu số $Z(x) = \sum_{y'} \pi_{\text{ref}}(y' \mid x) \exp(r(x, y')/\beta)$ đòi hỏi phải cộng tổng trên **tất cả mọi câu trả lời $y'$ có thể có trong vũ trụ** — một con số lớn vô tận không máy tính nào tính nổi!
  * **Cú lật ngược phương trình của DPO (Giải phương trình Slide 17 để rút $r(x, y)$ theo $\pi^*$):**
    Lấy $\log$ hai vế của công thức $\pi^*(y \mid x)$ và chuyển vế, ta biểu diễn được hàm thưởng $r(x, y)$ thông qua chính sách $\pi^*$:
    $$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$
  * **Lắp $r(x, y)$ vào hiệu số của Mô hình So sánh Cặp Bradley-Terry (Công thức 3):**
    $$r(x, y_w) - r(x, y_l) = \left[ \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \cancel{\beta \log Z(x)} \right] - \left[ \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \cancel{\beta \log Z(x)} \right]$$
* **Lời giảng giải:**
  * Hãy nhìn vào dòng cuối cùng của Slide 18 — đây chính là khoảnh khắc đẹp nhất của toán học Post-Training:
  * Đại lượng bất khả thi $\beta \log Z(x)$ chỉ phụ thuộc vào câu hỏi $x$ mà **không phụ thuộc vào câu trả lời $y_w$ hay $y_l$**. Do đó, khi đặt câu thắng $y_w$ và câu thua $y_l$ của **cùng một câu hỏi $x$** vào hiệu số Bradley-Terry $r(x, y_w) - r(x, y_l)$, hai số hạng $\beta \log Z(x)$ trừ cho nhau **bằng $0$ tuyệt đối**! Cản trở lớn nhất của toán học đã tự động bốc hơi!

---

#### **Slide 19: Toán học từ RLHF $\to$ DPO (Bước 3/3): Hàm Mất mát DPO & Đạo hàm Tự điều tốc**
* **Nội dung trên Slide:**
  * **Công thức Cốt lõi số 5 (Hàm mất mát Direct Preference Optimization — DPO):**
    Thay hiệu số vừa rút gọn ở Slide 18 vào hàm mất mát Bradley-Terry (Công thức 3), ta thu được:
    $$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \log \sigma\left( \underbrace{\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}}_{\text{Phần thưởng ngầm định } \hat{r}_\theta(x, y_w)} - \underbrace{\beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}}_{\text{Phần thưởng ngầm định } \hat{r}_\theta(x, y_l)} \right) \right]$$
  * **Đạo hàm (Gradient) của DPO giải thích cơ chế học:**
    $$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = -\beta \, \mathbb{E}_{(x, y_w, y_l)} \Big[ \underbrace{\sigma\big(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\big)}_{\text{Bộ điều tốc sai số}} \cdot \Big( \underbrace{\nabla_\theta \log \pi_\theta(y_w \mid x)}_{\text{Kéo } y_w \text{ lên}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l \mid x)}_{\text{Đẩy } y_l \text{ xuống}} \Big) \Big]$$
* **Lời giảng giải:**
  * Qua đúng 3 bước biến đổi đại số (Slide 17 $\to$ 18 $\to$ 19), chúng ta đã chứng minh được rằng: việc tối ưu hóa hàm mất mát đơn giản $\mathcal{L}_{\text{DPO}}(\theta)$ ở Công thức 5 **tương đương về mặt toán học** với toàn bộ quy trình 2 giai đoạn (Huấn luyện Reward Model + Chạy RL phạt KL)!
  * Nhìn vào đạo hàm của DPO: Mô hình đồng thời **kéo xác suất câu thắng $y_w$ lên** và **đẩy xác suất câu thua $y_l$ xuống**, được điều tiết bởi **bộ điều tốc $\sigma(\hat{r}_l - \hat{r}_w)$** (tự động phạt mạnh khi mô hình đang xếp hạng sai, và tự động giảm lực cập nhật về $0$ khi mô hình đã xếp hạng đúng).

---

#### **Slide 20: Sơ đồ Kiến trúc & Luồng Dữ liệu của DPO (Chỉ cần 2 Mô hình, Không cần RL)**
* **Nội dung trên Slide:**
  * **Sơ đồ Luồng Hoạt động của DPO (So sánh trực diện với Sơ đồ 4 mô hình của RLHF ở Slide 15):**
    ```text
                                 ┌──► [ 1. Mô hình Gốc π_ref ] (Đóng băng) ──► Tính log π_ref(y_w|x) và log π_ref(y_l|x) ──┐
                                 │                                                                                          ▼
    Dữ liệu Cặp có sẵn ──────────┤                                                                           Tính hiệu điểm thưởng ngầm định:
    (x, Câu thắng y_w, Câu thua y_l)                                                                         Δr̂ = β log(π_θ(y_w)/π_ref(y_w))
                                 │                                                                              - β log(π_θ(y_l)/π_ref(y_l))
                                 └──► [ 2. Mô hình Đang học π_θ ] (Huấn luyện) ──► Tính log π_θ(y_w|x) và log π_θ(y_l|x) ───┘
                                                 ▲                                                                          │
                                                 └──────────── Cập nhật Gradient trực tiếp từ L_DPO = -log σ(Δr̂) ◄──────────┘
    ```
* **Lời giảng giải:**
  * Đặt sơ đồ DPO ở Slide 20 cạnh sơ đồ RLHF+PPO ở Slide 15, người nghe sẽ thấy ngay sự tối giản vượt bậc:
    * **Không còn Reward Model $r_\phi$!**
    * **Không còn mạng Critic $V_\psi$!**
    * **Không còn bước chờ mô hình tự gõ từng chữ (Autoregressive Rollout)!** Toàn bộ câu thắng $y_w$ và câu thua $y_l$ đã có sẵn trong tập dữ liệu nên được đưa thẳng vào mô hình trong 1 lượt tính toán song song giống hệt như SFT!

---

#### **Slide 21: [CHUYỂN GIAO 4] Giới hạn của DPO Gốc $\longrightarrow$ Động cơ ra đời các Biến thể (SimPO, RPO, IPO, KTO)**
* **Nội dung trên Slide:**
  * **[3 GIỚI HẠN CỦA DPO GỐC TRONG LÝ THUYẾT & THỰC NGHIỆM]:**
    1. **Thiên kiến Độ dài (*Length Bias*):** Vì $\log \pi_\theta(y \mid x) = \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$ là tổng trên toàn bộ các từ, câu trả lời càng dài (nhiều từ $|y|$) thì tổng hiệu số càng phình to $\to$ Mô hình sau khi học DPO bị bệnh **nói dài dòng lê thê**.
    2. **Tụt Xác suất Câu đúng (*Likelihood Displacement*):** Khi $y_w$ và $y_l$ giống nhau ở 90% phần đầu câu, việc đạp mạnh câu sai $y_l$ xuống vô tình kéo tụt luôn xác suất tuyệt đối của cả câu đúng $y_w$.
    3. **Đói dữ liệu cặp:** Đòi hỏi bắt buộc phải có đúng 2 câu $(y_w, y_l)$ cho cùng một câu hỏi $x$.
  * **$\Longrightarrow$ ĐỘNG CƠ RA ĐỜI CÁC BIẾN THỂ NÂNG CẤP CỦA DPO (2024–2025):**
    * **SimPO:** Chia trung bình cho độ dài câu $\frac{1}{|y|} \log \pi_\theta(y \mid x)$ để trị bệnh nói dài dòng và bỏ luôn $\pi_{\text{ref}}$.
    * **RPO (DPO + NLL):** Cộng thêm một mỏ neo SFT nhỏ $-\alpha \log \pi_\theta(y_w \mid x)$ để giữ xác suất câu đúng không bao giờ bị tụt.
    * **IPO:** Dùng hàm mất mát bình phương để dừng cập nhật ngay khi đạt khoảng cách mục tiêu, chống Overfitting.
    * **KTO:** Học trực tiếp từ dữ liệu đơn lẻ (chỉ biết câu đó được bấm *Like* hay *Dislike*) mà không cần cặp so sánh.

---

#### **Slide 22: [CHUYỂN GIAO 5] Giới hạn của Học Sở thích Tĩnh (*Offline DPO*) $\longrightarrow$ Động cơ quay lại Học Tăng cường (*Online RL*)**
* **Nội dung trên Slide:**
  * **Câu hỏi lớn:** *"Nếu DPO đơn giản và hiệu quả như vậy, tại sao khi làm các mô hình suy luận giải Toán/Code (như OpenAI o1 hay DeepSeek-R1), người ta lại phải quay về dùng Học Tăng cường Trực tuyến (Online RL)?"*
  * **[GIỚI HẠN CỐT LÕI CỦA OFFLINE DPO — SỰ LỆCH PHÂN PHỐI & THIẾU KHÁM PHÁ]:**
    * DPO học trên một tập dữ liệu cặp $(y_w, y_l)$ **tĩnh (cố định từ trước)**.
    * Khi mô hình $\pi_\theta$ thay đổi dần qua quá trình học, cách suy nghĩ của nó đã khác xa tập dữ liệu ban đầu, nhưng nó không hề được tự viết thử để biết suy nghĩ mới của mình là đúng hay sai (**Thiếu khả năng tự khám phá — No Online Exploration**).
  * **$\Longrightarrow$ ĐỘNG CƠ SANG NHÁNH B (RL & REASONING):** Để giải quyết các bài toán suy luận sâu, mô hình bắt buộc phải **tự mình thử nghiệm sinh câu trả lời mới (Online Rollout)** và học trực tiếp từ phần thưởng!

---

### PHẦN 3 (NHÁNH B): HỌC TĂNG CƯỜNG & NÂNG CAO SUY LUẬN — POLICY GRADIENT, PPO, RLVR & GRPO *(Slides 23 – 36)*

#### **Slide 23: Tổng quan Nhánh B (*Reinforcement Learning & Reasoning Enhancement*)**
* **Nội dung trên Slide:**
  * **Câu hỏi trung tâm:** *"Làm thế nào để mô hình tự nâng cao năng lực suy luận thông qua việc tự thử nghiệm các câu trả lời và nhận phần thưởng phản hồi?"*
  * **Hai mảnh ghép độc lập trong Nhánh B:**
    * **B1. Thiết lập RL & Thiết kế Phần thưởng (*RL Setup & Reward Design*):** Mô hình lấy điểm thưởng từ đâu? (**RLHF** cảm tính vs. **RLVR** kiểm chứng khách quan; Chấm điểm kết quả cuối **ORM** vs. Chấm điểm từng bước **PRM**).
    * **B2. Thuật toán Tối ưu hóa Chính sách (*Policy Optimization*):** Dùng công thức toán học nào để biến điểm thưởng thành bước cập nhật trọng số? (**Policy Gradient**, **PPO**, **GRPO**).

---

#### **Slide 24: B1. Sơ đồ Chu trình Học Tăng Cường (Online RL Setup) cho LLM**
* **Nội dung trên Slide:**
  * **Sơ đồ Chu trình Khép kín của Online RL:**
    ```text
                        ┌─────────────────────────────────────────────────────────────────┐
                        ▼                                                                 │
    Câu hỏi x ──► [ Chính sách (Policy π_θ) ] ──► Tự sinh từng từ ──► Câu trả lời y = (y_1, ..., y_T)
                        ▲                                                                 │
                        │                                                                 ▼
                [ Cập nhật Trọng số θ ] ◄──── Điểm thưởng R(x, y) ◄──── [ Hệ thống Chấm điểm / Môi trường ]
             (Tăng xác suất nếu R cao,                                   (Reward Model hoặc Bộ kiểm chứng)
              Giảm xác suất nếu R thấp)
    ```
* **Lời giảng giải:**
  * Nhìn vào sơ đồ: Điểm khác biệt sống còn của Nhánh B so với SFT và Offline DPO là mũi tên **"Tự sinh từng từ"** nằm ngay trong vòng lặp huấn luyện! Mô hình không học từ văn bản có sẵn trên đĩa cứng, mà tự thử nghiệm giải bài toán bằng chính năng lực hiện tại của nó, nhận điểm thưởng $R(x, y)$, rồi cập nhật trọng số $\theta$ ngay lập tức.

---

#### **Slide 25: Nguyên lý Toán học Cốt lõi của RL — *Policy Gradient* & Đường cơ sở (*Baseline*)**
* **Nội dung trên Slide:**
  * **Công thức Cốt lõi số 6 (Đạo hàm Chính sách *Policy Gradient* với Hàm Lợi thế *Advantage*):**
    $$\nabla_\theta \mathcal{J}_{\text{RL}}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \Big[ \underbrace{\big( R(x, y) - b(x) \big)}_{\text{Lợi thế } A(x, y)} \cdot \nabla_\theta \log \pi_\theta(y \mid x) \Big]$$
  * **Tại sao bắt buộc phải trừ đi Đường cơ sở $b(x)$ (Điểm kỳ vọng trung bình)?**
    * Nếu không có $b(x)$ và thang điểm thưởng luôn dương (ví dụ từ $1$ đến $10$ điểm): Một câu trả lời kém đạt $2$ điểm vẫn mang dấu dương $(+2)$, nên đạo hàm vẫn kéo xác suất của câu trả lời kém đó tăng lên!
    * Khi trừ đi điểm trung bình $b(x)$ (ví dụ $b(x) = 6$ điểm):
      * Câu đạt $9$ điểm có Lợi thế $A = 9 - 6 = \mathbf{+3 > 0}$ $\longrightarrow$ Được **tăng xác suất**!
      * Câu đạt $2$ điểm có Lợi thế $A = 2 - 6 = \mathbf{-4 < 0}$ $\longrightarrow$ Bị **giảm xác suất**!

---

#### **Slide 26: [CHUYỂN GIAO 6] Giới hạn của RLHF trong Suy luận $\longrightarrow$ Động cơ ra đời RLVR (*Verifiable Rewards*)**
* **Nội dung trên Slide:**
  * **[GIỚI HẠN CỦA NEURAL REWARD MODEL (RLHF) TRONG TOÁN HỌC & LẬP TRÌNH]:**
    * Trong các bài toán suy luận chặt chẽ (Toán, Code, Logic), một mạng nơ-ron giám khảo $r_\phi(x, y)$ thường xuyên chấm điểm sai vì bản thân mạng giám khảo cũng không biết giải bài toán khó!
    * Mô hình chính sách $\pi_\theta$ sẽ dễ dàng đánh lừa giám khảo $r_\phi$ bằng cách viết một lời giải ngụy biện trình bày đẹp mắt, nhiều công thức nhưng kết quả cuối cùng sai hoàn toàn (*Reward Hacking*).
  * **$\Longrightarrow$ ĐỘNG CƠ RA ĐỜI RLVR (*RL with Verifiable Rewards — Lambert et al., 2024; DeepSeek-R1, 2025*):**
    * Vứt bỏ hoàn toàn mạng nơ-ron Reward Model! Thay bằng **Bộ kiểm chứng Quy tắc Tất định (*Deterministic Verifier*)**:
      * *Toán học:* Trích xuất đáp số cuối cùng so khớp với đáp án chuẩn (Đúng $= 1$, Sai $= 0$).
      * *Lập trình:* Chạy đoạn code qua trình biên dịch và bộ kiểm thử Unit Tests (Qua hết test $= 1$, Lỗi $= 0$).
    * **Ưu điểm lý thuyết tuyệt đối:** Chân lý khách quan 100%, **miễn nhiễm hoàn toàn với *Reward Hacking***!

---

#### **Slide 27: [CHUYỂN GIAO 7] Giám sát Quá trình (*PRM*) vs. Giám sát Kết quả (*ORM*)**
* **Nội dung trên Slide:**
  * **Hai cách Gán Tín dụng (*Credit Assignment*) cho một bài giải dài nhiều bước:**
    * **PRM (*Process Reward Model*):** Chấm điểm đúng/sai cho **từng dòng lập luận trung gian**.
    * **ORM (*Outcome Reward Model*):** Chỉ chấm điểm đúng/sai tại **đáp số cuối cùng**.
  * **[GIỚI HẠN CỦA PRM $\longrightarrow$ ĐỘNG CƠ DÙNG ORM THUẦN TÚY TRONG DEEPSEEK-R1]:**
    * *3 giới hạn khiến PRM thất bại ở quy mô lớn:* (1) Không thể định nghĩa ranh giới thế nào là "1 bước", (2) PRM lại phải dùng mạng nơ-ron chấm từng bước nên lại bị *Reward Hacking*, và (3) PRM phạt ngay các bước "thử nghiệm sai tạm thời", vô tình giết chết khả năng thử–và–sai của mô hình!
    * *Động cơ dùng ORM:* Chỉ chấm điểm kết quả cuối cùng giúp giải phóng hoàn toàn quá trình suy nghĩ trung gian — mô hình được tự do thử nghiệm, tự phát hiện lỗi và tự sửa miễn là ra đáp số cuối đúng!

---

#### **Slide 28: B2. Thuật toán Tối ưu hóa số 1 — PPO (*Proximal Policy Optimization*) & Công thức *PPO-Clip***
* **Nội dung trên Slide:**
  * **Hai trụ cột toán học của PPO (Schulman et al., 2017):**
    1. Dùng **Mạng Critic $V_\psi(s_t)$** dự đoán điểm kỳ vọng tại từng vị trí từ thứ $t$ để tính lợi thế $\hat{A}_t$.
    2. Dùng **Hàm cắt tỉa Vùng tin cậy (*PPO-Clip*)** để ngăn mô hình thay đổi xác suất quá đột ngột trong 1 bước cập nhật:
  * **Công thức Cốt lõi số 7 (Hàm mục tiêu Cắt tỉa *PPO-Clip*):**
    $$\mathcal{L}_{\text{PPO-CLIP}}(\theta) = \mathbb{E}_t \Big[ \min\Big( \rho_t(\theta) \hat{A}_t, \; \text{clip}\big(\rho_t(\theta), \, 1 - \epsilon, \, 1 + \epsilon\big) \hat{A}_t \Big) \Big], \qquad \text{với } \rho_t(\theta) = \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\theta_{\text{old}}}(y_t \mid s_t)}$$
* **Lời giảng giải:**
  * Công thức 7 giải quyết một rủi ro chí mạng của RL: Nếu ở một bước cập nhật, mô hình thay đổi xác suất $\pi_\theta$ quá xa so với $\pi_{\theta_{\text{old}}}$ (tỷ số $\rho_t$ quá lớn hoặc quá nhỏ), mô hình sẽ bị trượt khỏi vùng ngôn ngữ chuẩn và sập hoàn toàn. Hàm $\text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)$ đóng vai trò như chiếc phanh an toàn: chỉ cho phép xác suất mỗi từ thay đổi tối đa $\pm 20\%$ ($\epsilon = 0.2$) trong một vòng lặp!

---

#### **Slide 29: [CHUYỂN GIAO 8] Giới hạn của PPO trong Suy luận Chuỗi dài $\longrightarrow$ Động cơ ra đời GRPO**
* **Nội dung trên Slide:**
  * **[2 GIỚI HẠN CHÍ MẠNG CỦA PPO KHI HUẤN LUYỆN MÔ HÌNH SUY LUẬN (REASONING)]:**
    1. **Gánh nặng Mạng Critic $V_\psi$:** Mạng Critic có kích thước khổng lồ ngang ngửa mô hình ngôn ngữ chính, làm tăng gấp đôi chi phí bộ nhớ và tính toán.
    2. **Sự bất lực của Critic trên chuỗi suy nghĩ dài:** Khi mô hình viết chuỗi nháp `<think>` dài 10,000 từ mà điểm thưởng chỉ có ở đáp số cuối cùng, mạng Critic $V_\psi(s_t)$ rất khó dự đoán đúng điểm kỳ vọng ở từ thứ 500 hay từ thứ 2,000, gây ra sai số lớn trong lợi thế $\hat{A}_t$.
  * **$\Longrightarrow$ ĐỘNG CƠ RA ĐỜI GRPO (*Group Relative Policy Optimization — Shao et al., 2024*):**
    * *"Làm thế nào để tính được Đường cơ sở $b(x)$ và Lợi thế $A(x, y)$ một cách chính xác mà **xóa bỏ hoàn toàn mạng Critic $V_\psi$**?"*

---

#### **Slide 30: Sơ đồ Kiến trúc & Luồng Dữ liệu của RLVR + GRPO (Không cần Critic!)**
* **Nội dung trên Slide:**
  * **Sơ đồ Luồng Hoạt động của GRPO (So sánh trực diện với Sơ đồ PPO ở Slide 15):**
    ```text
                                           ┌──► Câu trả lời o_1: "<think>... </think> Đáp án: 19" ──► Bộ kiểm chứng ──► r_1 = 1.0 ──┐
                                           ├──► Câu trả lời o_2: "<think>... </think> Đáp án: 17" ──►   (Verifier   ──► r_2 = 0.0 ──┤
    Câu hỏi x ──► [ Policy π_θ ] ──► Sinh ─┼──► Câu trả lời o_3: "<think>... </think> Đáp án: 15" ──►   Toán/Code)  ──► r_3 = 0.0 ──┼─► Tính Trung bình μ_r
                  (Lấy mẫu Nhóm G)   Nhóm  └──► Câu trả lời o_G: "<think>... </think> Đáp án: 19" ──►               ──► r_G = 1.0 ──┘   & Độ lệch chuẩn σ_r
                        ▲                                                                                                                     │
                        │                                                                                                                     ▼
                        └──────────────────── Cập nhật Gradient (PPO-Clip + Phạt KL so với π_ref) ◄──── Lợi thế Nhóm: Â_i = (r_i - μ_r) / σ_r ┘
    ```
* **Lời giảng giải:**
  * Nhìn vào sơ đồ của GRPO: Cả **Reward Model mạng nơ-ron** lẫn **mạng Critic $V_\psi$** của PPO đều đã biến mất!
  * Thay vào đó, với 1 câu hỏi $x$, mô hình $\pi_\theta$ tự sinh ra một **Nhóm gồm $G$ câu trả lời độc lập** $(o_1, \dots, o_G)$. Bộ kiểm chứng chấm điểm khách quan cho từng câu thu được $(r_1, \dots, r_G)$. Chính điểm trung bình $\mu_r$ của nhóm $G$ câu trả lời này đóng vai trò là đường cơ sở $b(x)$ để các câu trả lời trong nhóm tự so sánh với nhau!

---

#### **Slide 31: Công thức Toán học của GRPO & Minh họa Bằng Con số Cụ thể**
* **Nội dung trên Slide:**
  * **Công thức Cốt lõi số 8 (Lợi thế Tương đối Nhóm của GRPO — *Group Relative Advantage*):**
    $$\hat{A}_i = \frac{r_i - \text{mean}(r_1, r_2, \dots, r_G)}{\text{std}(r_1, r_2, \dots, r_G)}$$
  * **Ví dụ tính nhẩm với nhóm $G = 4$ câu trả lời:**
    * Giả sử $r_1 = 1$ (Đúng) và $r_2 = r_3 = r_4 = 0$ (Sai).
    * Điểm trung bình nhóm: $\text{mean}(r) = 0.25$; Độ lệch chuẩn: $\text{std}(r) \approx 0.43$.
    * Câu đúng $o_1$ nhận lợi thế: $\hat{A}_1 = \frac{1 - 0.25}{0.43} = \mathbf{+1.73 > 0}$ *(Toàn bộ các từ trong $o_1$ được kéo xác suất tăng lên)*.
    * Ba câu sai $o_{2,3,4}$ nhận lợi thế: $\hat{A}_{2,3,4} = \frac{0 - 0.25}{0.43} = \mathbf{-0.58 < 0}$ *(Toàn bộ các từ trong 3 câu sai bị đẩy xác suất giảm xuống)*.

---

#### **Slide 32: Cơ chế Tự Phát sinh Chuỗi Suy luận `<think>` & Hiện tượng *"Aha Moment"* (*DeepSeek-R1*)**
* **Nội dung trên Slide:**
  * **Cách RLVR + GRPO đánh thức Tư duy Hệ thống 2 (*System 2 Reasoning*):**
    * Mô hình được yêu cầu đặt quá trình suy nghĩ nháp vào thẻ `<think> ... </think>` trước khi viết đáp án cuối.
    * Trong nhóm $G$ lần thử, những lần thử nào mà trong thẻ `<think>` mô hình tình cờ **biết dừng lại kiểm tra kết quả, phát hiện mâu thuẫn và quay lui tính lại (*"Wait, let me recheck..."*)** sẽ có xác suất ra đáp số cuối đúng ($r_i = 1 \implies \hat{A}_i > 0$) cao hơn hẳn những lần kết luận vội vàng!
    * Qua hàng nghìn bước cập nhật GRPO, các mẫu hành vi tự phản biện nhận được lợi thế dương $\hat{A}_i > 0$ liên tục và tự động trở thành chiến lược suy luận chủ đạo của mô hình (**Emergent Reasoning / Aha Moment**)!

---

#### **Slide 33: Hai Tinh chỉnh Lý thuyết của GRPO Hiện đại (*Dr. GRPO* & *DAPO*) & *Test-Time Compute***
* **Nội dung trên Slide:**
  * **Hai bài học lý thuyết thú vị khi tối ưu hóa GRPO (2025):**
    1. **Tại sao chia trung bình theo độ dài câu $\frac{1}{|o_i|}$ trong GRPO gốc lại làm câu trả lời SAI ngày càng dài ra? (*Dr. GRPO*):** Vì khi câu trả lời sai ($\hat{A}_i < 0$), mô hình cố tình viết lan man thật dài để mẫu số $|o_i|$ lớn lên nhằm pha loãng hình phạt âm trên mỗi từ! *(Khắc phục: Chia cho hằng số cố định $L_{\max}$)*.
    2. **Tại sao cần nới rộng trần cắt tỉa phía trên $\epsilon_{\text{high}}$? (*DAPO*):** Để các từ khóa mở đầu cho sự phản biện (như từ *"Wait"*) vốn ban đầu có xác suất rất nhỏ có thể tăng tốc bứt phá nhanh hơn mà không bị phanh PPO-Clip chặn đứng sớm.
  * **Mở rộng Tính toán lúc Chạy (*Test-Time Compute*):** Cho phép mô hình suy nghĩ lâu hơn hoặc lấy mẫu nhiều phương án rồi bỏ phiếu đa số (*Majority Voting*).

---

#### **Slide 34: [CHUYỂN GIAO 9] Giới hạn của RL trên Mô hình Nhỏ $\longrightarrow$ Động cơ ra đời Chưng cất (*Distillation*)**
* **Nội dung trên Slide:**
  * **Câu hỏi thực tế:** *"Nếu RLVR + GRPO tạo ra khả năng suy luận kỳ diệu như vậy trên mô hình lớn (671 tỷ tham số), ta có nên lấy một mô hình nhỏ (7B hoặc 32B) cho tự chạy RL từ con số 0 không?"*
  * **[GIỚI HẠN CỦA TỰ CHẠY RL TRÊN MÔ HÌNH NHỎ — BÀI TOÁN MÒ KIM ĐÁY BỂ]:**
    * Nhìn lại Công thức 8 của GRPO: Nếu với một bài toán khó, mô hình nhỏ thử cả $G = 16$ lần đều giải sai hết ($r_1 = r_2 = \dots = r_{16} = 0$), thì điểm trung bình bằng $0$ và mọi lợi thế $\hat{A}_i = 0$ — **mô hình hoàn toàn không học được bất cứ điều gì**!
    * Vì dung lượng bộ não của mô hình nhỏ hạn hẹp, nó không thể tự mò ra được lời giải đúng trong không gian tìm kiếm khổng lồ của chuỗi suy nghĩ dài.
  * **$\Longrightarrow$ ĐỘNG CƠ SANG NHÁNH C (MODEL DISTILLATION):** Thay vì bắt mô hình nhỏ tự mò mẫm bằng RL, hãy dùng mô hình lớn (đã qua RL) làm **Thầy giáo (Teacher)** truyền thụ trực tiếp năng lực xuống mô hình **Học trò (Student)**!

---

### PHẦN 4 (NHÁNH C): CHƯNG CẤT MÔ HÌNH (*MODEL DISTILLATION*) & TỔNG KẾT TOÀN CẢNH *(Slides 35 – 45)*

#### **Slide 35: Tổng quan Nhánh C (*Model Distillation*) — Học Thầy–Trò**
* **Nội dung trên Slide:**
  * **Câu hỏi trung tâm:** *"Làm thế nào để một mô hình Học trò (Student — nhỏ, nhẹ, nhanh) kế thừa được năng lực từ một mô hình Thầy (Teacher — khổng lồ, uyên bác)?"*
  * **Bằng chứng thực nghiệm từ *DeepSeek-R1 (2025)*:**
    * Mô hình 32B học **Chưng cất (Distillation)** từ mô hình Thầy 671B đạt **72.6%** điểm thi toán AIME, đè bẹp hoàn toàn mô hình 32B tự chạy RL từ con số 0 (chỉ đạt **47.0%**), với chi phí huấn luyện rẻ hơn hàng chục lần!

---

#### **Slide 36: Sơ đồ Ba Hình thức Chuyển giao Tri thức trong Chưng cất (*Forms of Transferred Knowledge*)**
* **Nội dung trên Slide:**
  * **Sơ đồ Phân loại 3 Cấp độ Chưng cất theo Loại Thông tin Truyền thụ:**
    ```text
    Câu hỏi x ──► [ MÔ HÌNH THẦY (Teacher) ]
                        │
                        ├──► 1. Chưng cất Dự đoán (Logit / Soft-target KD):
                        │       Truyền bảng phân phối xác suất mềm P_Teacher(v | s_t) ở từng bước từ
                        │
                        ├──► 2. Chưng cất Đầu ra (Sequence / Response KD):
                        │       Truyền câu trả lời văn bản hoàn chỉnh y^(Teacher)
                        │
                        └──► 3. Chưng cất Suy luận (Reasoning / Rationale KD):
                                Truyền cả chuỗi suy nghĩ trung gian "<think> Các bước lập luận... </think> Đáp án"
                                        │
                                        ▼
                            [ MÔ HÌNH TRÒ (Student π_θ) ] (Học tái tạo lại dự đoán / đầu ra / suy luận của Thầy)
    ```
* **Lời giảng giải:**
  * Sơ đồ này bám sát cấu trúc Nhánh C trong `post_training_tree_outline.md`: tùy thuộc vào việc ta lấy thông tin gì từ mô hình Thầy để dạy cho mô hình Trò, ta có 3 phương pháp chưng cất với đặc tính lý thuyết khác nhau.

---

#### **Slide 37: Hình thức 1 — Chưng cất Dự đoán (*Logit Distillation*) & Khái niệm *"Tri thức Tối"***
* **Nội dung trên Slide:**
  * **Công thức Cốt lõi số 9 (Làm mềm Phân phối bằng Nhiệt độ $\tau$ & Chưng cất Logit — Hinton et al., 2015):**
    $$P^{(\tau)}(\text{từ } v \mid s_t) = \frac{\exp(z_v / \tau)}{\sum_{w \in \mathcal{V}} \exp(z_w / \tau)}, \qquad \mathcal{L}_{\text{Logit-KD}}(\theta) = D_{\text{KL}}\Big( P_{\text{Teacher}}^{(\tau)}(\cdot \mid s_t) \;\Big\|\; \pi_{\text{Student}}^{(\tau)}(\cdot \mid s_t) \Big)$$
  * **Ý nghĩa của "Tri thức Tối" (*Dark Knowledge*):**
    * Khi dự đoán từ tiếp theo cho câu *"Con mèo đang nằm ngủ trên..."*, nhãn văn bản thông thường chỉ cho biết từ được chọn là `"thảm"` ($100\%$).
    * Nhưng phân phối xác suất mềm của Thầy (nhờ nhiệt độ làm mềm $\tau > 1$) tiết lộ thêm cấu trúc quan hệ giữa mọi từ vựng: `["thảm": 70%, "ghế": 29.9%, "đám_mây": 0.0001%]`. Học theo phân phối mềm này giúp Trò nắm bắt được ngữ nghĩa sâu sắc nhanh gấp nhiều lần!

---

#### **Slide 38: Hình thức 2 & 3 — Chưng cất Đầu ra (*Sequence KD*) vs. Chưng cất Suy luận (*Rationale KD*)**
* **Nội dung trên Slide:**
  * **Khi nào không dùng được Chưng cất Logit (Hình thức 1)?**
    * Khi mô hình Thầy là API đóng (không trả về bảng xác suất toàn từ vựng) hoặc khi Thầy và Trò dùng hai bộ từ vựng (Tokenizer) khác nhau.
  * **Sự tiến hóa từ Hình thức 2 sang Hình thức 3:**
    * *Chưng cất Đầu ra (Sequence KD — Kim & Rush, 2016):* Chỉ dạy Trò bắt chước đáp án cuối cùng của Thầy $\to$ Với bài toán khó, Trò chỉ học vẹt bề mặt mà không hiểu logic.
    * *Chưng cất Suy luận (Rationale / Step-by-Step KD — Hsieh et al., 2023; DeepSeek-R1, 2025):* Cho Thầy sinh ra cả **chuỗi suy nghĩ trung gian `<think>`** (lọc qua bộ kiểm chứng để loại bỏ câu sai), rồi dạy Trò học toàn bộ quá trình lập luận từng bước đó!

---

#### **Slide 39: [CHUYỂN GIAO 10] Giới hạn của *Forward KL* $\longrightarrow$ Động cơ dùng *Reverse KL* trong Chưng cất**
* **Nội dung trên Slide:**
  * **Công thức Cốt lõi số 10 (Hai chiều Phân kỳ KL khi truyền thụ phân phối từ Thầy $P_T$ sang Trò $\pi_\theta$):**
    $$\text{Forward KL (KD cổ điển): } D_{\text{KL}}(P_T \parallel \pi_\theta) = \sum_{y} P_T(y \mid x) \log \frac{P_T(y \mid x)}{\pi_\theta(y \mid x)}$$
    $$\text{Reverse KL (MiniLLM / GKD): } D_{\text{KL}}(\pi_\theta \parallel P_T) = \sum_{y} \pi_\theta(y \mid x) \log \frac{\pi_\theta(y \mid x)}{P_T(y \mid x)}$$
  * **[GIỚI HẠN CỦA FORWARD KL $\longrightarrow$ ĐỘNG CƠ DÙNG REVERSE KL]:**
    * *Giới hạn của Forward KL (Mean-Seeking):* Bất cứ nơi nào Thầy có $P_T(y \mid x) > 0$, Trò bị ép phải có $\pi_\theta(y \mid x) > 0$. Vì Thầy rất lớn nên biết 10 cách trả lời đúng khác nhau; Trò dung lượng nhỏ không nhớ xuể cả 10 cách, bị ép phải kéo giãn phân phối ra giữa các đỉnh, dẫn đến sinh câu trả lời nửa nạc nửa mỡ!
    * *Động cơ dùng Reverse KL (Mode-Seeking):* Cho phép Trò gán $\pi_\theta = 0$ ở 9 cách trả lời phụ mà không bị phạt ($0 \log 0 = 0$), giúp Trò dồn 100% dung lượng bộ não nhỏ bé để làm chủ **1 cách trả lời xuất sắc nhất** của Thầy!

---

#### **Slide 40: Mở rộng Lý thuyết Không gian Tham số — Thích nghi Hạng thấp (*LoRA*) & Hợp nhất Mô hình (*Model Merging*)**
* **Nội dung trên Slide:**
  * **Giả thuyết Chiều Nội tại Thấp (*Intrinsic Dimensionality — LoRA*):**
    * Sự thay đổi trọng số $\Delta \mathbf{W}$ trong quá trình Post-training thực chất chỉ diễn ra trên một không gian con có số chiều rất nhỏ ($\Delta \mathbf{W} = \frac{\alpha}{r}\mathbf{B}\mathbf{A}$ với $r \ll d$). Vì đóng băng trọng số gốc $\mathbf{W}_0$, LoRA tuân theo quy luật lý thuyết: *"Học ít hơn nhưng cũng quên kiến thức gốc ít hơn"*.
  * **Số học Vector Tác vụ (*Task Vectors & Model Merging — TIES / DARE*):**
    * Hiệu số trọng số $\tau_k = \theta_k - \theta_{\text{base}}$ biểu diễn một vector kỹ năng trong không gian tham số. Ta có thể kết hợp nhiều mô hình chuyên biệt (Toán, Code, Ngôn ngữ) xuất phát từ cùng một $\theta_{\text{base}}$ bằng cách lọc nhiễu và cộng trực tiếp các vector kỹ năng mà không cần huấn luyện lại!

---

#### **Slide 41: Lý thuyết Đánh giá Mô hình — Thuế Căn chỉnh (*Alignment Tax*) & Thiên kiến Trọng tài AI**
* **Nội dung trên Slide:**
  * **Thuế Căn chỉnh (*Alignment Tax*) & Quên Thảm khốc (*Catastrophic Forgetting*):**
    * Khi ép mô hình quá chặt vào một khuôn mẫu hành vi hoặc an toàn, độ đa dạng và một phần năng lực gốc có thể bị suy giảm.
  * **Đánh giá Đối kháng bằng Hệ số Elo (*Chatbot Arena*):**
    * Áp dụng trực tiếp mô hình Bradley-Terry (Công thức 3) trên các trận đấu cặp ẩn danh để xếp hạng các mô hình.
  * **3 Thiên kiến Lý thuyết của Trọng tài AI (*LLM-as-a-Judge Biases*):**
    1. *Thiên kiến vị trí (Position Bias):* Ưu ái câu trả lời đứng trước (khử bằng cách đảo vị trí chấm 2 lần).
    2. *Thiên kiến độ dài (Verbosity Bias):* Ưu ái câu trả lời dài dòng (khử bằng hồi quy kiểm soát độ dài *Length-Controlled Evaluation*).
    3. *Thiên kiến tự ưu ái (Self-Enhancement Bias):* Thích văn phong của mô hình cùng họ.

---

#### **Slide 42: Phân biệt Rõ các Khái niệm Chồng lấn trong Hệ sinh thái Post-Training**
* **Nội dung trên Slide:**
  * **Sơ đồ Phân tầng Khái niệm (Tránh nhầm lẫn thuật ngữ):**
    ```text
    [ TẦNG 1: NGUỒN TÍN HIỆU THƯỞNG ]      [ TẦNG 2: CÁCH GÁN TÍN DỤNG ]      [ TẦNG 3: THUẬT TOÁN TỐI ƯU HÓA ]
    • RLHF (Sở thích Con người)            • ORM (Chấm kết quả cuối cùng)     • PPO  (RL có mạng Critic + Clip)
    • RLAIF (Sở thích của AI)        ───►  • PRM (Chấm từng bước suy luận) ─► • GRPO (RL chuẩn hóa nhóm, bỏ Critic)
    • RLVR (Kiểm chứng Toán / Code)                                           • DPO  (Học trực tiếp, không qua RL)
    ```
* **Lời giảng giải:**
  * Nhìn vào 3 cột trên slide, mọi sự nhầm lẫn về thuật ngữ đều được giải tỏa:
    * Tổ hợp kinh điển cho Chatbot hội thoại mở: **RLHF (Tầng 1) + ORM (Tầng 2) + PPO hoặc DPO (Tầng 3)**.
    * Tổ hợp kinh điển cho Mô hình Suy luận Toán/Code (như DeepSeek-R1): **RLVR (Tầng 1) + ORM (Tầng 2) + GRPO (Tầng 3)**.

---

#### **Slide 43: Sơ đồ Toàn cảnh Chuỗi Tiến hóa Lý thuyết (Từ SFT $\to$ RLHF $\to$ DPO $\to$ GRPO $\to$ Distillation)**
* **Nội dung trên Slide:**
  ```text
  [Pre-training]
       │ (Giới hạn: Chỉ biết điền tiếp văn bản, không biết trả lời yêu cầu)
       ▼
     [SFT] 
       │ (Giới hạn: Exposure Bias, Học vẹt gây ảo giác, Không biết câu trả lời sai)
       ▼
  [RLHF + PPO] ──(Giới hạn: Cồng kềnh 4 mô hình, phải học Reward Model riêng)──► [DPO & SimPO]
       │                                                                          (Giải nghiệm đóng,
       │ (Giới hạn khi giải Toán/Code:                                             chỉ cần 2 mô hình,
       │  • Neural RM bị lừa điểm Reward Hacking                                   nhưng học tĩnh Offline)
       │  • Mạng Critic của PPO quá nặng trên chuỗi <think> dài)
       ▼
  [RLVR + GRPO] (Thay RM bằng Bộ kiểm chứng khách quan + Thay Critic bằng Trung bình Nhóm G câu)
       │
       │ (Giới hạn: Mô hình nhỏ tự chạy RL không mò ra được lời giải đúng để nhận thưởng)
       ▼
  [Model Distillation] (Chưng cất Dự đoán Logits / Đầu ra / Chuỗi suy luận <think> từ Thầy sang Trò)
  ```

---

#### **Slide 44: Bảng Đối chiếu Tổng kết Ba Trụ cột Post-Training**
* **Nội dung trên Slide:**

| Trụ cột Lý thuyết | Câu hỏi Trung tâm | Động cơ & Ý tưởng Cốt lõi | Phương pháp Tiêu biểu |
| :--- | :--- | :--- | :--- |
| **A. Tối ưu hóa Sở thích & Căn chỉnh** *(Preference & Alignment)* | *Thế nào là một câu trả lời tốt?* | Khắc phục điểm mù của SFT bằng cách học từ so sánh cặp $(y_w \succ y_l)$ dưới ràng buộc phạt KL. | Reward Modeling (Bradley-Terry), RLHF, RLAIF, DPO, SimPO |
| **B. Học Tăng cường & Suy luận** *(RL & Reasoning)* | *Làm sao học từ thử nghiệm & phần thưởng?* | Khắc phục *Exposure Bias* và sự tĩnh tại của Offline bằng cách tự sinh câu trả lời và tối ưu Lợi thế $A = R - b$. | Policy Gradient, RLVR, ORM / PRM, PPO, GRPO |
| **C. Chưng cất Mô hình** *(Model Distillation)* | *Làm sao chuyển giao năng lực sang mô hình nhỏ?* | Khắc phục giới hạn khám phá của mô hình nhỏ bằng cách học từ xác suất mềm và chuỗi `<think>` của Thầy. | Logit KD (Dark Knowledge), Sequence KD, Rationale KD (Reverse KL) |

---

#### **Slide 45: 5 Thông điệp Nền tảng Đọng lại (*Take-Home Messages*) & Phiên Thảo luận (Q&A)**
* **Nội dung trên Slide:**
  1. **Mọi phương pháp mới đều sinh ra từ giới hạn của phương pháp cũ:** Từ *Pre-training* (lệch mục tiêu) $\to$ *SFT* (lệch phân phối & thiếu tín hiệu âm) $\to$ *RLHF* (cồng kềnh 4 mô hình) $\to$ *DPO* (thiếu khám phá trực tuyến) $\to$ *RLVR/GRPO* (quá khó cho mô hình nhỏ) $\to$ *Distillation*.
  2. **10 Công thức Toán học Cốt lõi** xoay quanh một nguyên lý thống nhất: **Dịch chuyển xác suất $\log \pi_\theta(y \mid x)$ theo hướng có lợi thế cao hơn kỳ vọng, dưới sự kìm hãm của phân kỳ KL**.
  3. **DPO** là hệ quả đại số trực tiếp của việc giải nghiệm đóng bài toán RLHF rồi triệt tiêu hằng số $Z(x)$ nhờ phép trừ cặp Bradley-Terry.
  4. **RLVR + ORM + GRPO** là công thức nền tảng đánh thức khả năng tự suy luận (`<think>` / *Aha Moment*) mà không cần mạng giám khảo cảm tính hay mạng Critic.
  5. **Model Distillation** (đặc biệt là Chưng cất Suy luận và *Reverse KL*) là cầu nối đưa trí thông minh của các mô hình khổng lồ về các mô hình nhỏ gọn.
