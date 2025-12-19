%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import importlib
from matplotlib.lines import Line2D
import time

# Import sub-modules
import cell_decomposition
import node_generation
import path_planning
import reduced_graph
import aco_mmas
import input_processing

# Reload để đảm bảo cập nhật code mới nhất
importlib.reload(cell_decomposition)
importlib.reload(node_generation)
importlib.reload(path_planning)
importlib.reload(reduced_graph)
importlib.reload(aco_mmas)
importlib.reload(input_processing)

from cell_decomposition import boustrophedon_decomposition
from node_generation import node_generation
from path_planning import full_coverage_planner
from reduced_graph import build_reduced_graph, build_distance_matrix, astar, reconstruct_path
from aco_mmas import MMAS
from input_processing import cad_to_ogm

class CoveragePipeline:
    def __init__(self, charging_station=(0, 0)):
        self.charging_station = tuple(map(int, charging_station))

        # Dữ liệu bản đồ & Vùng
        self.grid = None
        self.cells = None
        self.cell_map = None
        self.processed_cells = None

        # Dữ liệu Đồ thị & ACO
        self.graph = None
        self.dist_matrix = None
        self.best_sequence = None

        # Kết quả Lộ trình & Metrics
        self.detailed_path = None
        self.metrics = {}

        # Đo runtime
        self.runtimes = {}

    # --- STAGE 0: INPUT ---
    def process_input_cad(self, image_path, grid_size=(100, 100), fill_closed_regions=False):
        start = time.time()
        print("[Step 1] Input Processing...")
        self.grid = cad_to_ogm(
            image_path,
            grid_size=grid_size,
            fill_closed_regions_flag=fill_closed_regions,
        )
        self.runtimes['Step 1: Preprocessing'] = time.time() - start
        return self.grid

    # --- STAGE 1: DECOMPOSITION (SPLITTED) ---
    def run_cell_decomposition(self):
        """Tách biệt Boustrophedon Decomposition"""
        start = time.time()
        print("[Step 2] Decomposing Cells...")
        self.cells, self.cell_map = boustrophedon_decomposition(self.grid)
        self.runtimes['Step 2: BCD Decomposition'] = time.time() - start
        return self.cells

    def run_node_generation(self):
        """Tách biệt Trích xuất thuộc tính Node"""
        start=time.time()
        print("[Step 3] Extracting Node Information...")
        self.processed_cells = node_generation(self.cells, self.grid)
        self.runtimes['Step 3: Node Generation'] = time.time() - start
        return self.processed_cells

    # --- STAGE 2: SEQUENCE OPTIMIZATION (SPLITTED) ---
    def build_graph(self):
        """Xây dựng đồ thị kết nối giữa các Cell"""
        start=time.time()
        print("[Step 4] Building Reduced Graph...")
        self.graph = build_reduced_graph(self.grid, self.processed_cells)
        self.runtimes['Step 4: Graph Building'] = time.time() - start
        return self.graph

    def build_distance_matrix(self):
        """Tính toán ma trận khoảng cách từ đồ thị"""
        start=time.time()
        print("[Step 5] Calculating Distance Matrix...")
        K = len(self.processed_cells)
        self.dist_matrix = build_distance_matrix(self.graph, K, self.grid, self.processed_cells)
        # Xử lý vô cùng (vùng bị cô lập)
        self.dist_matrix[np.isinf(self.dist_matrix)] = 9999.0
        self.runtimes['Step 5: Distance Matrix'] = time.time() - start
        return self.dist_matrix

    def run_aco(self, ants=20, iters=50):
        """Chạy thuật toán Kiến để tìm chuỗi tối ưu"""
        start=time.time()
        print("[Step 6] Running ACO Optimization...")
        # Tìm node bắt đầu gần trạm sạc
        dist_to_start = [np.linalg.norm(np.array(self.charging_station) - np.array(d['centroid']))
                         for d in self.processed_cells.values()]
        start_node = np.argmin(dist_to_start)

        solver = MMAS(
            self.dist_matrix, 
            num_ants=ants, 
            num_iterations=iters, 
            beta=4.0, 
            closed_tour=False
        )
        
        self.best_sequence, self.best_cost = solver.run(
            start_node=start_node, 
            verbose=True, 
            early_stopping=20
        )
        self.runtimes['Step 6: ACO Pathfinding'] = time.time() - start
        return self.best_sequence

    # --- STAGE 3: PATH PLANNING & PERFORMANCE ---
    def generate_final_path(self):
        start=time.time()
        print("[Step 7] Generating Final Path...")
        _, self.detailed_path, _ = full_coverage_planner(
            self.processed_cells, self.best_sequence, self.grid, self.charging_station
        )
        self.runtimes['Step 7: Trajectory Planning'] = time.time() - start
        print("Performance Analysis...")
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Tính toán chi tiết các chỉ số Performance nâng cao."""
        all_pts = []
        clean_steps = 0
        trans_steps = 0

        for seg in self.detailed_path:
            pts = seg['points']
            all_pts.extend(pts)
            if seg['type'] == 'zigzag':
                clean_steps += len(pts)
            else:
                trans_steps += len(pts)

        # 1. Cơ bản
        self.metrics['total_steps'] = len(all_pts)
        self.metrics['clean_area_steps'] = clean_steps
        self.metrics['transition_steps'] = trans_steps

        # 2. Coverage & Overlap
        free_space_total = np.sum(self.grid == 0)
        visited_unique_set = set([tuple(map(int, p)) for p in all_pts])
        visited_unique_count = len(visited_unique_set)
        
        self.metrics['coverage_rate'] = (visited_unique_count / free_space_total) * 100
        
        # Công thức Overlap: (Tổng bước - Bước duy nhất) / Bước duy nhất
        if visited_unique_count > 0:
            overlap = ((len(all_pts) - visited_unique_count) / visited_unique_count) * 100
            self.metrics['overlap_rate'] = overlap
        else:
            self.metrics['overlap_rate'] = 0

        # 3. Turns (90 and 180)
        t90, t180 = 0, 0
        for i in range(1, len(all_pts) - 1):
            v1 = np.array(all_pts[i]) - np.array(all_pts[i-1])
            v2 = np.array(all_pts[i+1]) - np.array(all_pts[i])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_theta = np.dot(v1, v2) / (n1 * n2)
                angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                if 80 <= angle <= 100: t90 += 1
                elif angle > 170: t180 += 1

        self.metrics['turns_90'] = t90
        self.metrics['turns_180'] = t180
        
        # 4. Angular Momentum (Tổng góc quay lũy kế)
        self.metrics['total_heading_change'] = (t90 * 90) + (t180 * 180)

        # 5. Energy Index (Dựa trên trọng số ma sát và cơ cấu truyền động)
        # Đi thẳng: 1.0, Quay 90: 2.5, Quay 180: 4.0
        self.metrics['energy_index'] = (len(all_pts) * 1.0) + (t90 * 2.5) + (t180 * 4.0)

        # 6. Connectivity Success
        # Vì ta đã lọc vùng cô lập trong Decomposition, tỷ lệ này sẽ là 100% nếu robot hoàn thành chuỗi
        self.metrics['connectivity_success'] = 100.0 if len(self.best_sequence) > 0 else 0.0

        #Tính độ mượt (càng thấp càng mượt)
        smoothness_index = self.metrics['total_heading_change'] / self.metrics['total_steps']
        self.metrics['smoothness_index'] = smoothness_index

        # Tính mật độ rẽ
        turn_density = (t90 + t180) / self.metrics['total_steps']
        self.metrics['turn_density'] = turn_density

    def _show_runtime_report(self):
        total_time = sum(self.runtimes.values())
        print("\n" + "⏱️ " + "═"*43)
        print(f"{'SYSTEM RUNTIME PROFILE':^45}")
        print("─"*45)
        for stage, duration in self.runtimes.items():
            percentage = (duration / total_time) * 100
            print(f"🔹 {stage:<28} : {duration:>7.4f}s ({percentage:>5.1f}%)")
        print("─"*45)
        print(f"🚀 TOTAL PIPELINE RUNTIME      : {total_time:>7.4f}s")
        print("═"*45 + "\n")
        self.metrics['total_runtime'] = total_time

    def _show_report(self):
        self._show_runtime_report()
        m = self.metrics
        
        print("\n" + "=" * 60)
        print(f"{'📊 ADVANCED PERFORMANCE REPORT':^60}")
        print("=" * 60)

        def row(icon, label, value, unit=""):
            # Tách biểu tượng (icon) ra khỏi label để căn lề chữ chính xác hơn
            # Căn lề nhãn 32 ký tự, giá trị căn phải 10 ký tự
            print(f" {icon} {label:<32} | {value:>10} {unit}")

        si = m['smoothness_index']
        
        # Đánh giá độ mượt theo thang đo kỹ thuật
        if si < 12: 
            status = "Excellent"
        elif si < 20: 
            status = "Good"
        else: 
            status = "Fair"

        print("\n" + "=" * 65)
        print(f"{'📊 ADVANCED PERFORMANCE REPORT':^65}")
        print("=" * 65)

        # Nhóm 1: Coverage & Path
        row("📍", "Coverage Rate", f"{m['coverage_rate']:.2f}", "%")
        row("♻️", "Overlap Rate", f"{m['overlap_rate']:.2f}", "%")
        row("🚀", "Total Steps", m['total_steps'], "steps")
        row("🧹", "Cleaning Steps", m['clean_area_steps'], "steps")
        row("🔗", "Transition Steps", m['transition_steps'], "steps")
        
        print("-" * 65)
        
        # Nhóm Smoothness (Căn chỉnh thủ công để khớp với hàm row)
        print(f" 🧩 {'Path Smoothness Index':<32} | {si:>10.2f} °/step")
        print(f" 📈 {'Smoothness Quality':<32} | {status:>10}")
        print(f" 📍 {'Turn Density':<32} | {m['turn_density']:>10.3f} turns/step")
        
        print("-" * 65)

        # Nhóm 2: Maneuverability
        row("↪️", "90° Turns (L-turn)", m['turns_90'])
        row("🔄", "180° Turns (U-turn)", m['turns_180'])
        row("📐", "Total Heading Change", m['total_heading_change'], "degrees")
        
        print("-" * 65)

        # Nhóm 3: Efficiency & Energy
        efficiency = m['clean_area_steps'] / m['total_steps'] if m['total_steps'] > 0 else 0
        row("📊", "Efficiency Ratio", f"{efficiency:.3f}")
        row("⚡", "Estimated Energy Index", f"{m['energy_index']:.1f}", "units")
        row("✅", "Connectivity Success", f"{m['connectivity_success']:.1f}", "%")

        print("=" * 65 + "\n")

    def print_mission_report(self):
        """
        In báo cáo chi tiết lộ trình dọn dẹp dưới dạng văn bản.
        Hiển thị thứ tự Step, Cell ID, loại vùng, hướng quét và chi phí di chuyển.
        """
        if not self.best_sequence or self.dist_matrix is None:
            print("❌ Lỗi: Pipeline chưa chạy xong. Vui lòng chạy ACO trước khi in báo cáo.")
            return

        print("\n" + "╔" + "═"*77 + "╗")
        print(f"║ {'MISSION SUMMARY: FULL COVERAGE PATH PLANNING':^75} ║")
        print("╠" + "═"*77 + "╣")
        print(f"║ {'Step':<6} | {'Cell ID':<8} | {'Type':<14} | {'Orientation':<12} | {'Travel Dist':<12} | {'Progress':<8} ║")
        print("╟" + "─"*77 + "╢")

        total_cells = len(self.best_sequence)
        total_deadheading = 0

        for i, cid in enumerate(self.best_sequence):
            cell_data = self.processed_cells[cid]
            cell_size = len(cell_data['coordinates'])

            # 1. Phân loại Cell
            c_type = "Large Room" if cell_size > 500 else "Small Corner"

            # 2. Hướng quét (Lấy từ node_generation)
            orient = cell_data.get('orientation', 'N/A').capitalize()

            # 3. Tính chi phí di chuyển (Travel Cost/Deadheading)
            travel_cost = 0
            if i > 0:
                prev_cid = self.best_sequence[i-1]
                # Lấy khoảng cách từ ma trận khoảng cách mà ACO đã sử dụng
                travel_cost = self.dist_matrix[prev_cid][cid]
                total_deadheading += travel_cost

            # 4. Tiến độ (%)
            progress = f"{(i+1)/total_cells:>6.1%}"

            print(f"║ {i+1:<6} | {f'Cell {cid}':<8} | {c_type:<14} | {orient:<12} | {travel_cost:<12.1f} | {progress:<8} ║")

        print("╚" + "═"*77 + "╝")

        # In các thông số tổng quan quan trọng
        print(f"📊 [TOTAL METRICS]")
        print(f" - Total Cells to Clean    : {total_cells}")
        print(f" - Unproductive Travel     : {total_deadheading:.1f} pixels (Lower is better)")
        print(f" - Algorithm Efficiency    : {self.runtimes.get('Step 6: ACO Pathfinding', 0):.2f}s (ACO Runtime)")
        print("="*87 + "\n")

    def visualize(self):
        print("--- Stage 4: Visualizing Result ---")
        plt.figure(figsize=(25, 25))

        # 1. Hiển thị Map và Cells
        plt.imshow(self.grid, cmap='binary', origin='upper')
        plt.imshow(self.cell_map, cmap='Set3', alpha=0.3, origin='upper')

        # 2. Vẽ Charging Station
        plt.scatter(self.charging_station[1], self.charging_station[0], marker='p',
        
                    color='gold', s=300, edgecolors='black', linewidth=2, label='Charging Station', zorder=10)
        plt.text(self.charging_station[1], self.charging_station[0] - 2, "HOME / CHARGER",
                 color='darkgoldenrod', weight='bold', ha='center', fontsize=10, zorder=10)

        # 3. Vẽ lộ trình chi tiết
        for segment in self.detailed_path:
            pts = np.array(segment['points'])
            if segment['type'] == 'zigzag':
                # Đường dọn dẹp màu Cyan
                plt.plot(pts[:, 1], pts[:, 0], color='#00f2ff', linewidth=1, zorder=2)

                # Điểm vào (Entry - Tam giác xanh) và điểm ra (Exit - Tròn cam)
                plt.scatter(segment['entry'][1], segment['entry'][0], marker='>',
                            color='lime', s=80, edgecolors='black', zorder=5)
                plt.scatter(segment['exit'][1], segment['exit'][0], marker='o',
                            color='orange', s=80, edgecolors='black', zorder=5)
            else:
                # Đường nối A* màu đỏ nét đứt
                plt.plot(pts[:, 1], pts[:, 0], color='red', linewidth=2, linestyle='--', alpha=0.7, zorder=4)

        # 4. Đánh số thứ tự STEP (Cell)
        for i, cid in enumerate(self.best_sequence):
            r, c = self.processed_cells[cid]['centroid']

            # Badge hình tròn với màu sắc tương phản (DarkOrange/DeepPink)
            # Màu này sẽ nổi bật hoàn toàn trên nền Cyan/Binary map
            plt.text(c, r, f"{i+1}",
                     color='white',
                     weight='bold',
                     fontsize=8,
                     ha='center',
                     va='center',
                     bbox=dict(
                         facecolor='#e65100', # Màu cam đậm (Deep Orange) - cực kỳ nổi bật trên Cyan
                         alpha=0.9,          # Tăng độ đậm để che bớt đường zigzag bên dưới nhãn
                         edgecolor='white',
                         boxstyle='circle,pad=0.2',
                         linewidth=1         # Viền trắng giúp tách biệt khỏi nền
                     ),
                     zorder=10) # Đẩy zorder lên cao nhất để không bị đường nào đè qua

            # Thu nhỏ ID Cell và làm mờ hơn nữa để tránh rối
            plt.text(c, r + 5, f"c{cid}",
                     color='#333333',
                     fontsize=6,
                     alpha=0.4,
                     fontstyle='italic',
                     ha='center',
                     va='top',
                     zorder=9)
        # 5. CHÈN THÔNG TIN RUNTIME VÀO GÓC BẢN ĐỒ
        total_time = self.metrics.get('total_runtime', 0)
        coverage = self.metrics.get('coverage_rate', 0)

        # Thay thế Emoji bằng Text tiêu chuẩn hoặc ký tự đặc biệt được hỗ trợ
        stats_text = (f"Runtime: {total_time:.3f}s\n"
                      f"Coverage: {coverage:.1f}%")

        # Đặt text ở góc (điều chỉnh tọa độ dựa trên GRID_SIZE của bạn)
        # Với map 200x200, y=190 là gần mép dưới
        plt.text(5, 192, stats_text, color='white', fontsize=11, fontweight='bold',
                 bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'),
                 zorder=15)

        # 6. Chú thích (Legend) - Đã dời ra ngoài theo ý bạn
        custom_lines = [
            Line2D([0], [0], color='#00f2ff', lw=3),
            Line2D([0], [0], color='red', lw=3, linestyle='--'),
            Line2D([0], [0], marker='>', color='lime', markersize=12, linestyle='None'),
            Line2D([0], [0], marker='o', color='orange', markersize=12, linestyle='None')
        ]

        plt.legend(
            custom_lines,
            ['Dọn dẹp (Zigzag)', 'Di chuyển (A*)', 'Entry (Vào)', 'Exit (Ra)'],
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            title="Ký hiệu bản đồ",
            shadow=True
        )

        centroid_sequence = []
        for cid in self.best_sequence:
            centroid_sequence.append(self.processed_cells[cid]['centroid'])

        c_rows, c_cols = zip(*centroid_sequence)
        # Vẽ đường nối các tâm cell (màu đỏ đô, nét đứt, mờ) để thấy luồng di chuyển chính
        plt.plot(c_cols, c_rows, color='red', linestyle='--', linewidth=1, alpha=0.3, zorder=5, label='Flow Order')

        # --- ĐÁNH SỐ THỨ TỰ & GRADIENT ---
        num_steps = len(self.best_sequence)
        for i, cid in enumerate(self.best_sequence):
            r, c = self.processed_cells[cid]['centroid']

            # Tính toán độ đậm dựa trên tiến độ (Gradient Alpha)
            # Bước càng về sau alpha càng cao (từ 0.4 đến 1.0)
            step_alpha = 0.4 + (i / num_steps) * 0.6

            # Badge Step hình tròn
            plt.text(c, r, f"{i+1}",
                     color='white', weight='bold', fontsize=8,
                     ha='center', va='center',
                     bbox=dict(
                         facecolor='#e65100',
                         alpha=step_alpha,
                         edgecolor='white',
                         boxstyle='circle,pad=0.2',
                         linewidth=0.5
                     ),
                     zorder=10)

        plt.title(f"Complete Coverage Path Planning (Map: {self.grid.shape})", fontsize=15, pad=20)
        plt.tight_layout()
        plt.grid(True, which='both', linestyle=':', alpha=0.3)
        plt.savefig("coverage_result.png", dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # 1. Cấu hình tham số
    IMAGE_PATH = "test_3.jpg"  # File ảnh Sofa bạn đã gửi
    GRID_SIZE = (200, 200)           # Kích thước lưới (nên từ 300-500 cho bản đồ này)
    CHARGING_STATION = (5, 30)      # Tọa độ trạm sạc (y, x)

    pipeline = CoveragePipeline(charging_station=CHARGING_STATION)

    # Bắt đầu đo tổng thời gian thực thi
    overall_start = time.time()

    # Chạy tuần tự các bước
    pipeline.process_input_cad(IMAGE_PATH, grid_size=GRID_SIZE,fill_closed_regions=False)
    pipeline.run_cell_decomposition()
    pipeline.run_node_generation()
    pipeline.build_graph()
    pipeline.build_distance_matrix()
    pipeline.run_aco(ants=20, iters=80) # Tăng ants/iters theo map 200x200
    pipeline.generate_final_path()

    # In báo cáo chi tiết
    pipeline._show_report()
    pipeline.print_mission_report()
    # Hiển thị kết quả
    pipeline.visualize()

    # Hiển thị kết quả cuối cùng
    plt.show()
  
