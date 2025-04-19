# Hệ thống Giao dịch Tự động Thông minh
# Tài liệu Hướng dẫn Cài đặt và Sử dụng

## Giới thiệu

Hệ thống Giao dịch Tự động Thông minh là một nền tảng giao dịch tiên tiến sử dụng kết hợp Ensemble Learning và Reinforcement Learning để tối ưu hóa chiến lược giao dịch. Hệ thống bao gồm các module chính sau:

1. **Module Thu thập và Xử lý Dữ liệu**: Thu thập dữ liệu thị trường từ nhiều nguồn khác nhau và tính toán các chỉ báo kỹ thuật.
2. **Module Ensemble Learning**: Kết hợp nhiều mô hình học máy để dự đoán xu hướng thị trường.
3. **Module Reinforcement Learning**: Tối ưu hóa chiến lược giao dịch dựa trên dự đoán xu hướng.
4. **Module Trading**: Thực hiện giao dịch tự động dựa trên chiến lược đã được tối ưu hóa.
5. **Dashboard**: Giao diện người dùng trực quan để theo dõi và quản lý hệ thống.
6. **Chat AI**: Tương tác với hệ thống bằng ngôn ngữ tự nhiên.

## Yêu cầu Hệ thống

- Docker và Docker Compose
- Tối thiểu 8GB RAM
- Tối thiểu 50GB dung lượng ổ đĩa
- Kết nối internet ổn định

## Cài đặt

### Bước 1: Cài đặt Docker và Docker Compose

Nếu bạn chưa cài đặt Docker và Docker Compose, hãy làm theo hướng dẫn tại:
- Docker: https://docs.docker.com/get-docker/
- Docker Compose: https://docs.docker.com/compose/install/

### Bước 2: Tải mã nguồn

```bash
git clone https://github.com/your-organization/trading-system.git
cd trading-system
```

Hoặc giải nén file nén đã được cung cấp:

```bash
unzip trading-system.zip
cd trading-system
```

### Bước 3: Cấu hình

Kiểm tra và chỉnh sửa file `.env` nếu cần thiết để phù hợp với môi trường của bạn.

### Bước 4: Khởi động hệ thống

```bash
docker-compose up -d
```

Quá trình khởi động có thể mất vài phút khi lần đầu tiên chạy do cần tải các image Docker và cài đặt các phụ thuộc.

### Bước 5: Truy cập hệ thống

Sau khi tất cả các dịch vụ đã khởi động, bạn có thể truy cập Dashboard tại:

```
http://localhost
```

## Sử dụng Hệ thống

### Dashboard

Dashboard cung cấp giao diện trực quan để:

1. **Tổng quan**: Xem tổng quan về hiệu suất hệ thống, bao gồm các chỉ số quan trọng như tổng số bot, lợi nhuận, vốn, tỷ lệ thắng, và biểu đồ hiệu suất.

2. **Quản lý Bot**:
   - Xem danh sách các bot giao dịch
   - Tạo bot mới
   - Cấu hình và khởi động/dừng bot
   - Xem chi tiết hiệu suất của từng bot

3. **Backtest**:
   - Kiểm tra chiến lược trên dữ liệu lịch sử
   - Phân tích hiệu suất với các biểu đồ và báo cáo chi tiết
   - So sánh các chiến lược khác nhau

### Chat AI

Hệ thống Chat AI cho phép bạn tương tác với hệ thống bằng ngôn ngữ tự nhiên. Một số ví dụ về câu hỏi và lệnh bạn có thể sử dụng:

- "Giá Bitcoin hiện tại là bao nhiêu?"
- "Hiển thị danh mục đầu tư của tôi"
- "Tạo một bot giao dịch cho Ethereum"
- "Các bot của tôi đang hoạt động như thế nào?"
- "Phân tích thị trường Bitcoin"
- "Giúp tôi hiểu cách sử dụng hệ thống"

## Kiến trúc Hệ thống

Hệ thống được xây dựng theo kiến trúc microservices, với mỗi module là một dịch vụ độc lập:

1. **data_service (cổng 8001)**: Thu thập và xử lý dữ liệu thị trường
2. **model_service (cổng 8002)**: Ensemble Learning để dự đoán xu hướng
3. **rl_service (cổng 8003)**: Reinforcement Learning để tối ưu hóa chiến lược
4. **trading_service (cổng 8004)**: Thực hiện giao dịch tự động
5. **chat_service (cổng 8005)**: Xử lý tương tác ngôn ngữ tự nhiên
6. **frontend (cổng 80)**: Giao diện người dùng

Các dịch vụ cơ sở dữ liệu:
- PostgreSQL: Lưu trữ dữ liệu lịch sử và cấu hình
- Redis: Cache và dữ liệu thời gian thực
- MongoDB: Lưu trữ lịch sử chat và dữ liệu phi cấu trúc

## Quản lý Hệ thống

### Khởi động/Dừng Hệ thống

```bash
# Khởi động tất cả dịch vụ
docker-compose up -d

# Dừng tất cả dịch vụ
docker-compose down

# Khởi động lại một dịch vụ cụ thể
docker-compose restart <service_name>
```

### Xem Logs

```bash
# Xem logs của tất cả dịch vụ
docker-compose logs

# Xem logs của một dịch vụ cụ thể
docker-compose logs <service_name>

# Xem logs theo thời gian thực
docker-compose logs -f <service_name>
```

### Sao lưu Dữ liệu

Dữ liệu được lưu trữ trong các volume Docker. Để sao lưu:

```bash
# Sao lưu PostgreSQL
docker exec trading_postgres pg_dump -U trading trading_system > backup_postgres.sql

# Sao lưu MongoDB
docker exec trading_mongodb mongodump --username trading --password trading_password --db trading_system --out /dump
docker cp trading_mongodb:/dump ./mongodb_backup
```

## Xử lý Sự cố

### Vấn đề Kết nối

Nếu bạn không thể truy cập Dashboard:

1. Kiểm tra xem tất cả các dịch vụ đã khởi động chưa:
   ```bash
   docker-compose ps
   ```

2. Kiểm tra logs để xem lỗi:
   ```bash
   docker-compose logs frontend
   ```

### Vấn đề Hiệu suất

Nếu hệ thống chạy chậm:

1. Kiểm tra tài nguyên hệ thống:
   ```bash
   docker stats
   ```

2. Tăng tài nguyên cho Docker nếu cần thiết

### Khôi phục Hệ thống

Nếu hệ thống gặp sự cố nghiêm trọng:

1. Dừng tất cả dịch vụ:
   ```bash
   docker-compose down
   ```

2. Xóa tất cả volume để bắt đầu lại từ đầu (cẩn thận, điều này sẽ xóa tất cả dữ liệu):
   ```bash
   docker-compose down -v
   ```

3. Khởi động lại hệ thống:
   ```bash
   docker-compose up -d
   ```

## Liên hệ Hỗ trợ

Nếu bạn gặp vấn đề không thể giải quyết hoặc có câu hỏi, vui lòng liên hệ:

- Email: support@trading-system.com
- Điện thoại: +XX-XXX-XXX-XXXX
- Website: https://trading-system.com/support
