# Upload lên Overleaf

**2 file paper trong folder:**
- `iceba_astro.tex` = **bản TIẾNG VIỆT** (để cô đọc/duyệt) ← đặt làm Main document khi đọc.
- `iceba_astro_en.tex` = **bản TIẾNG ANH** (bản nộp chính thức) ← đặt làm Main khi nộp.

1. Nén **cả folder này** (`overleaf/`) thành `.zip`.
2. overleaf.com → New Project → **Upload Project** → chọn file `.zip`.
3. Menu → **Main document** = file muốn xem (`iceba_astro.tex` để đọc VN); Compiler = **pdfLaTeX**.
4. Bấm **Recompile** → ra PDF. (Dùng `thebibliography` thủ công → KHÔNG cần chạy BibTeX.)
   *Nếu dấu tiếng Việt lỗi: đổi Compiler sang XeLaTeX theo ghi chú đầu file `iceba_astro.tex`.*

## Cần điền trước khi nộp (trong iceba_astro.tex)
- Tên tác giả + đơn vị + email (`\author` / `\affil`, đầu file).
- Author contributions (phần Declarations).
- `\bibitem{ourcell}` — trích dẫn bài gốc đếm tế bào của nhóm.

## Lưu ý hình
`fig_pub_recovery.png` hiện là bản **toy-sim (offline)**. Sau khi chạy Kaggle trên JSON SDSS thật,
thay file này bằng hình thật (cùng tên) rồi recompile.
