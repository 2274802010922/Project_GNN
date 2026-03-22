Training Pipeline | Pipeline xử lý
<p align="center"> <img src="images/pipeline.png" width="700"> </p>

EN:
The pipeline includes dataset loading, feature extraction, graph construction, and Graph Neural Network training. The constructed graph consists of 99 nodes and 8 edges, indicating a sparse structure.

VI:
Pipeline bao gồm tải dữ liệu, trích xuất đặc trưng, xây dựng đồ thị và huấn luyện mô hình Graph Neural Network. Đồ thị gồm 99 node và 8 cạnh, thể hiện cấu trúc rất thưa.

Training Loss (Early Stage) | Loss giai đoạn đầu
<p align="center"> <img src="images/loss_early.png" width="600"> </p>

EN:
The loss decreases steadily during the early training stage, indicating stable learning behavior and effective gradient updates. The model quickly learns basic patterns from the data.

VI:
Loss giảm đều ở giai đoạn đầu, cho thấy quá trình học ổn định và cập nhật gradient hiệu quả. Mô hình nhanh chóng học được các đặc trưng cơ bản.

Training Loss (Late Stage) | Loss giai đoạn sau
<p align="center"> <img src="images/loss_late.png" width="600"> </p>

EN:
The loss continues to decrease and converges to approximately 0.14. This demonstrates strong model convergence and effective optimization, with the model fine-tuning its parameters in later epochs.

VI:
Loss tiếp tục giảm và hội tụ về khoảng 0.14. Điều này cho thấy mô hình hội tụ tốt và tối ưu hiệu quả, với việc tinh chỉnh tham số ở các epoch sau.

GNN Explainer Scores | Kết quả GNN Explainer
<p align="center"> <img src="images/explainer_scores.png" width="700"> </p>

EN:
The explainer identifies important nodes and edges. A subset of edges has significantly higher importance, while others contribute minimally. This shows that the model focuses on meaningful relationships and ignores irrelevant connections.

VI:
GNNExplainer xác định các node và cạnh quan trọng. Một số cạnh có độ quan trọng rất cao, trong khi các cạnh khác gần như không ảnh hưởng. Điều này cho thấy mô hình tập trung vào các mối quan hệ có ý nghĩa và loại bỏ nhiễu.

Explanation Graph | Đồ thị giải thích
<p align="center"> <img src="images/explanation_graph.png" width="600"> </p>

EN:
This graph highlights the most influential nodes contributing to the prediction. It improves model interpretability by showing how different nodes affect the output.

VI:
Đồ thị này làm nổi bật các node có ảnh hưởng lớn đến kết quả dự đoán, giúp tăng khả năng giải thích của mô hình.

Graph Visualization with Images | Đồ thị với dữ liệu ảnh
<p align="center"> <img src="images/graph_with_images.png" width="700"> </p>

EN:
Each node corresponds to a real image, allowing intuitive understanding of how visual data is structured within the graph. This bridges the gap between raw data and graph representation.

VI:
Mỗi node tương ứng với một ảnh thực, giúp người dùng dễ dàng hiểu cách dữ liệu được tổ chức trong đồ thị. Điều này kết nối trực tiếp giữa dữ liệu gốc và biểu diễn graph.

Interactive Graph | Đồ thị tương tác
<p align="center"> <img src="images/interactive_graph.png" width="700"> </p>

EN:
The interactive graph enables dynamic exploration of node relationships. Users can better understand graph structure and connectivity.

VI:
Đồ thị tương tác cho phép người dùng khám phá mối quan hệ giữa các node một cách trực tiếp, giúp hiểu rõ cấu trúc đồ thị.

Overall Analysis | Phân tích tổng thể

EN:
The model demonstrates strong performance despite the sparse graph structure. The training process is stable, and the final loss value indicates effective learning.

The GNNExplainer results show that the model selectively focuses on important edges, confirming its ability to capture meaningful structural relationships.

Visualization results further validate that the model successfully learns both feature-level and graph-level representations, while also providing interpretability.

VI:
Mô hình cho thấy hiệu suất tốt dù đồ thị rất thưa. Quá trình huấn luyện ổn định và giá trị loss cuối thấp chứng tỏ mô hình học hiệu quả.

Kết quả từ GNNExplainer cho thấy mô hình tập trung vào các cạnh quan trọng, chứng minh khả năng học được các mối quan hệ có ý nghĩa.

Các hình ảnh trực quan cũng xác nhận rằng mô hình học được cả đặc trưng và cấu trúc, đồng thời có khả năng giải thích kết quả.
