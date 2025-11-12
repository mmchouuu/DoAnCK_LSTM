<h1 align="center"><b>ĐỒ ÁN CUỐI KỲ</b></h1>
<h2 align="center"><b>MACHINE TRANSLATION EN→FR (Encoder–Decoder LSTM)</b></h2>

## I. Tổng quan
Đồ án cuối kỳ môn Xử lý ngôn ngữ tự nhiên - NLP: **Xây dựng mô hình Dịch Máy** từ tiếng Anh tiếng Pháp bằng kiến trúc **Encoder–Decoder sử dụng LSTM**.  
Dự án được hiện thực bằng  **Python, PyTorch, TorchText, SpaCy**, gồm 3 phần chính:

- Chuẩn bị & tiền xử lý dữ liệu song ngữ, xây dựng từ điển (vocabulary).
- Xây dựng và huấn luyện mô hình Seq2Seq LSTM cho bài toán EN→FR.
- Đánh giá mô hình bằng BLEU Score và phân tích các lỗi dịch.

Toàn bộ mã nguồn, mô hình, và báo cáo được tổ chức theo cấu trúc thư mục chuẩn trong repo này.
Báo cáo cuối cùng: `report/DoAnCK_LSTM.pdf.`

**Công cụ sử dụng:**  
Python, PyTorch, TorchText, SpaCy, NLTK, Matplotlib, YAML

---

## II. Thành viên & Phân công công việc
Nhóm gồm **2 thành viên**, mỗi người đều tham gia cả **code** lẫn **viết báo cáo**.

| Thành viên                       | Phần code chính                                            | Phần báo cáo chính                                            | Sản phẩm cần có                                                                              |
| -------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Người 1 – Data & Pipeline**    | Xử lý dữ liệu, tokenization, tạo vocab, DataLoader, config | Giới thiệu đề tài, mô tả dữ liệu & pipeline xử lý             | `data/prepare_dataset.py`, `data/dataloader_utils.py`, `vocab_*.pkl`, hình minh hoạ pipeline |
| **Người 2 – Model & Evaluation** | Encoder, Decoder, Seq2Seq, train loop, inference, BLEU     | Mô tả mô hình, huấn luyện, đánh giá & phân tích lỗi, tổng kết | `models/*.py`, `evaluation/*.py`, biểu đồ loss, bảng BLEU, ví dụ dịch                        |


---

## III. Cấu trúc folder

```plaintext
DoAnCK_LSTM/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── prepare_dataset.py
│   ├── dataloader_utils.py
│   ├── vocab_en.pkl
│   ├── vocab_fr.pkl
│   └── README.md
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── seq2seq.py
│   ├── train.py
│   ├── utils_training.py
│   └── README.md
├── evaluation/
│   ├── inference.py
│   ├── evaluate_bleu.py
│   ├── examples_translation.txt
│   ├── analysis_error.md
│   └── README.md
├── checkpoints/
│   └── best_model.pth
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_experiments.ipynb
│   └── 03_results_visualization.ipynb
├── report/
│   ├── DoAnCK_LSTM.docx
│   ├── DoAnCK_LSTM.pdf
│   └── figures/
├── config/
│   └── params.yaml
├── requirements.txt
├── README.md
└── .gitignore

```

---

## IV. Quy trình làm việc nhóm
### 1. Khởi tạo repository và nhánh chính

```bash
git init
git branch -M main
git remote add origin https://github.com/<username>/DoAnCK_LSTM.git
git push -u origin main

```

### 2. Tạo các nhánh làm việc
- Nhánh tích hợp chung:
```bash
git checkout -b dev
git push -u origin dev
```
- Nhánh riêng cho từng người:
```bash
git checkout -b data-pipeline     # Người 1
git push -u origin data-pipeline

git checkout -b model-eval        # Người 2
git push -u origin model-eval

```

### 3. Cách làm việc h

| Thành viên | Nhánh làm việc | Thư mục chính |
| ---------- | -------------- | ------------- |
| A          | `data-pipeline`    | `/data`       |
| B          | `model-eval`  | `/models`, `evaluation/`     |


```bash
git add .
git commit -m "Implement data preprocessing"
git push origin data-pipeline
```

- Mỗi người làm trên nhánh cá nhân của mình, commit/push đều đặn.
- Mở Pull Request từ nhánh cá nhân → dev (được review/QA).
- Khi dev ổn định, mở Pull Request từ dev → main để chốt.



```bash
# (người duyệt)
git checkout main
git merge --no-ff dev
git push origin main

```

---

## V. Quy trình viết báo cáo chung (1 file PDF)
| Mục / Phần báo cáo                 | Người phụ trách    | Nội dung chính                                               | Tệp làm việc (có thể)                      | Kết quả đưa vào repo                         |
| ---------------------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------ | -------------------------------------------- |
| Giới thiệu                         | **Người 1**        | Động cơ, mục tiêu, phạm vi đề tài                            | `report/draft_intro.docx` hoặc `.md`       | **Chỉ commit** `report/DoAnCK_LSTM.pdf`      |
| Bài toán                           | **Người 1**        | Phát biểu bài toán MT EN→FR, metric, ràng buộc               | `report/draft_problem.md`                  | (PDF tổng)                                   |
| Dữ liệu                            | **Người 1**        | Nguồn dữ liệu, thống kê, ví dụ                               | `report/draft_data.md`                     | (PDF tổng)                                   |
| Chuẩn bị dữ liệu *(6.1)*           | **Người 1**        | Tiền xử lý, tokenization, vocab, batching                    | `report/draft_preprocess.md`               | (PDF tổng)                                   |
| Mô hình & Huấn luyện *(6.2–6.3)*   | **Người 2**        | Encoder–Decoder LSTM, tham số, loss/optimizer, training loop | `report/draft_model_train.md`              | (PDF tổng)                                   |
| Đánh giá & Phân tích *(6.4–9)*     | **Người 2**        | BLEU, ví dụ dịch, phân tích lỗi                              | `report/draft_eval.md`                     | (PDF tổng)                                   |
| Kết luận & Hướng phát triển *(10)* | **Người 2**        | Tóm tắt kết quả, đề xuất cải tiến                            | `report/draft_conclusion.md`               | (PDF tổng)                                   |
| Tổng hợp & định dạng cuối          | 1 người (chủ biên) | Gom các phần, chuẩn hoá hình/bảng/trích dẫn                  | `report/DoAnCK_LSTM.docx` *(không commit)* | **Commit duy nhất** `report/DoAnCK_LSTM.pdf` |


## VI. Cách chạy dự án
### 1. Cài môi trường

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

### 2. Chuẩn bị dữ liệu
```bash
python data/prepare_dataset.py
```

### 3. Huấn luyện mô hình

```bash
python models/train.py
```

### 4. Kiểm thử dịch

```bash
python evaluation/inference.py "How are you today?"
```

### 5. Kiểm tra BLEU score

```bash
python evaluation/evaluate_bleu.py
```

## VII. Cấu hình mô hình 

```plaintext
File: config/params.yaml
```

```bash
embedding_dim: 512
hidden_size: 512
num_layers: 2
dropout: 0.3
teacher_forcing_ratio: 0.5
batch_size: 64
learning_rate: 0.001
num_epochs: 20
max_length: 50
```
## **VIII. Phụ lục**
### **1. requirements.txt**


```bash
torch>=2.3.0
torchtext>=0.18.0
spacy>=3.7.0
nltk>=3.9
matplotlib>=3.8
pyyaml>=6.0
tqdm>=4.66

```

### **2. .gitignore**

```bash
__pycache__/
*.pyc
*.pyo
*.pyd
*.ipynb_checkpoints/
.venv/
.env

checkpoints/
data/processed/
report/*.docx
report/figures/*.png

```

### **3. Tài liệu tham khảo**

- Sutskever et al. (2014). Sequence to Sequence Learning with Neural Networks.
- PyTorch Documentation: https://pytorch.org/docs/stable/nn.html#lstm

- Multi30K Dataset: https://github.com/multi30k/dataset


**Tác giả**: Nhóm DoAnCK_LSTM — Học kỳ 1 (2025–2026)
**Nộp bài**: 14/12/2025 — qua hệ thống E-Learning
