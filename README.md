### learn-deeplearning
Mình tự học Deep Learning cơ bản trong năm dịch thứ 2.

### I. Hồi quy tuyến tính (Linear Regression)
<br />
# Thuật toán linear regression giải quyết các bài toán có đầu ra là giá trị thực, ví dụ: dự đoán giá nhà, dự đoán giá cổ phiếu, dự đoán tuổi,...
<br />
Bài toán: 
Bạn làm ở công ty bất động sản, bạn có dữ liệu về diện tích và giá nhà, giờ có một ngôi nhà mới
bạn muốn ước tính xem giá ngôi nhà đó khoảng bao nhiêu. Trên thực tế thì giá nhà phụ thuộc rất
nhiều yếu tố: diện tích, số phòng, gần trung tâm thương mại,.. nhưng để cho bài toán đơn giản giả
sử giá nhà chỉ phụ thuộc vào diện tích căn nhà.
<br />

Bài toán quy về bài toán tìm giá trị nhỏ nhất của hàm số: 
J(w0;w1) = 1/2∗ (N∑i=1(w0 + w1 ∗ xi − yi)**2)
<br />

Việc tìm giá trị lớn nhất hàm này hoàn toàn có thể giải được bằng đại số nhưng để giới thiệu thuậttoán Gradient descent cho bài Neural network nên tôi sẽ áp dụng Gradient descent luôn.
Việc quan trọng nhất của thuật toán gradient descent là tính đạo hàm của hàm số nên giờ tasẽ đi tính đạo hàm theo từng biến.
Nhắc lại kiến thức h’(x) = f(g(x))’ = f’(g)*g’(x). Ví dụ:

<br />
h(x) = (3x + 1)2 thì f (x) = x2;g(x) = 3x + 1 => h0(x) = f 0(g) ∗ g0(x) = f 0(3x + 1) ∗ g0(x) =2 ∗ (3x + 1) ∗ 3 = 6 ∗ (3x + 1).
Tại 1 điểm (xi;yi) gọi f (w0;w1) = 1/2∗ (w0 + w1 ∗ xi − yi)**2
Ta có:

<br />
d f/dw0 = w0 + w1 ∗ xi − yi

<br />
d f/dw1 = xi ∗ (w0 + w1 ∗ xi − yi)

<br />
Do đó
dJ/dw0 = N∑i = 1 (w0 + w1 ∗ xi − yi)

<br />
dJ/dw1 = N∑i = 1 xi ∗ (w0 + w1 ∗ xi − yi)

<br />

![picture1](https://github.com/Dat0309/learn-deeplearning/blob/main/Linear%20regression/linear_regression.png)

<br />


### II.Logistic Regresstion

<br />
# thuật toánlogistic regression với đầu ra là giá trị nhị phân (0 hoặc 1), ví dụ: email gửi đến hòm thư của bạn có phải spam hay không; u là u lành tính hay ác tính,...

<br />

![picture2](https://github.com/Dat0309/learn-deeplearning/blob/main/Logictic%20Regression/logistic_regression.png)

### III.Neural Network
Neural là tính từ của neuron (nơ-ron), network chỉ cấu trúc đồ thị nên neural network (NN) là một hệ thống tính toán lấy cảm hứng từ sự hoạt động của các nơ-ron trong hệ thần kinh

<br />

![picture3](https://github.com/Dat0309/learn-deeplearning/blob/main/Neural%20Network/Neural_network.png)
