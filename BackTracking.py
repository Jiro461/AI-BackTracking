import heapq
import time
import tkinter as tk
from tkinter import ttk
import math
import random
from collections import deque

class TSPAlgorithms:
    def __init__(self, cities):
        self.cities = cities

    def dist(self, city1, city2):
        x1, y1 = self.cities[city1]
        x2, y2 = self.cities[city2]
        return math.ceil(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    def minimax(self, path, visited, depth, maximizing):
        """Minimax for TSP"""
        if len(path) == len(self.cities):  # All cities visited
            return self.calculate_total_cost(path + [path[0]]), path + [path[0]]

        if maximizing:
            best_value = -float('inf')
            best_path = None
            for i in range(len(self.cities)):
                if not visited[i]:
                    visited[i] = True
                    new_value, new_path = self.minimax(path + [i], visited, depth + 1, False)
                    if new_value > best_value:
                        best_value, best_path = new_value, new_path
                    visited[i] = False
            return best_value, best_path
        else:
            best_value = float('inf')
            best_path = None
            for i in range(len(self.cities)):
                if not visited[i]:
                    visited[i] = True
                    new_value, new_path = self.minimax(path + [i], visited, depth + 1, True)
                    if new_value < best_value:
                        best_value, best_path = new_value, new_path
                    visited[i] = False
            return best_value, best_path
        
    def simulated_annealing(self, start_city, initial_temp=1000, cooling_rate=0.99, stopping_temp=1):
        """Simulated Annealing for TSP"""
        start_time = time.time()
        import random

        def swap_random(path):
            i, j = random.sample(range(1, len(path) - 1), 2)  # Chỉ hoán đổi các thành phố giữa
            path[i], path[j] = path[j], path[i]

        def acceptance_probability(old_cost, new_cost, temperature):
            if new_cost < old_cost:
                return 1
            else:
                return math.exp((old_cost - new_cost) / temperature)

        n = len(self.cities)
        current_path = [start_city] + [i for i in range(n) if i != start_city] + [start_city]
        current_cost = self.calculate_total_cost(current_path)
        temp = initial_temp

        while temp > stopping_temp:
            new_path = current_path[:]
            swap_random(new_path)
            new_cost = self.calculate_total_cost(new_path)
            if acceptance_probability(current_cost, new_cost, temp) > random.random():
                current_path, current_cost = new_path, new_cost
            temp *= cooling_rate

        end_time = time.time()
        return current_path, current_cost, end_time - start_time
    
    def genetic_algorithm(self, population_size=100, generations=500, mutation_rate=0.05, start_city=0):
        """Genetic Algorithm for TSP with start city"""
        start_time = time.time()
        import random

        def initialize_population():
            population = []
            for _ in range(population_size):
                path = list(range(len(self.cities)))
                path.remove(start_city)
                random.shuffle(path)
                path.insert(0, start_city)  # Đặt thành phố bắt đầu vào vị trí đầu tiên
                population.append(path)
            return population

        def fitness(path):
            return 1 / self.calculate_total_cost(path + [path[0]])

        def selection(population):
            population.sort(key=lambda x: fitness(x), reverse=True)
            return population[:population_size // 2]

        def crossover(parent1, parent2):
            size = len(parent1)
            start, end = sorted(random.sample(range(1, size), 2))  # Bắt đầu từ vị trí 1 để không thay đổi thành phố bắt đầu
            child = [None] * size
            child[start:end] = parent1[start:end]
            pointer = end
            for city in parent2:
                if city not in child:
                    while child[pointer] is not None:
                        pointer = (pointer + 1) % size
                    child[pointer] = city
            return child

        def mutate(path):
            if random.random() < mutation_rate:
                i, j = random.sample(range(1, len(path)), 2)  # Đảm bảo không thay đổi thành phố bắt đầu
                path[i], path[j] = path[j], path[i]

        population = initialize_population()
        for _ in range(generations):
            selected = selection(population)
            population = selected[:]
            while len(population) < population_size:
                parent1, parent2 = random.sample(selected, 2)
                child = crossover(parent1, parent2)
                mutate(child)
                population.append(child)

        best_path = min(population, key=lambda x: self.calculate_total_cost(x + [x[0]]))
        best_cost = self.calculate_total_cost(best_path + [best_path[0]])
        end_time = time.time()
        return best_path + [best_path[0]], best_cost, end_time - start_time
    def solve_minimax(self, start_city):
            visited = [False] * len(self.cities)
            visited[start_city] = True
            start_time = time.time()
            cost, path = self.minimax([start_city], visited, 1, True)
            exec_time = time.time() - start_time
            return cost, path, exec_time
        
    def greedy(self, start_city):
        """Greedy Best First Search"""
        start_time = time.time()
        n = len(self.cities)
        visited = [False] * n
        path = [start_city]
        visited[start_city] = True
        total_cost = 0

        while len(path) < n:
            last_city = path[-1]
            next_city = min(
                [(i, self.dist(last_city, i)) for i in range(n) if not visited[i]],
                key=lambda x: x[1],
            )[0]
            visited[next_city] = True
            total_cost += self.dist(last_city, next_city)
            path.append(next_city)

        total_cost += self.dist(path[-1], path[0])  # Quay lại thành phố đầu
        end_time = time.time()
        return path + [path[0]], total_cost, end_time - start_time

    def hill_climbing(self, start_city):
        """Hill Climbing bắt đầu từ start_city"""
        start_time = time.time()
        n = len(self.cities)

        # Khởi tạo đường đi bắt đầu từ start_city
        path = [start_city] + [i for i in range(n) if i != start_city]
        path.append(start_city)  # Quay lại thành phố đầu
        current_cost = self.calculate_total_cost(path)

        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    new_path = path[:]
                    new_path[i:j] = reversed(new_path[i:j])
                    new_cost = self.calculate_total_cost(new_path)
                    if new_cost < current_cost:
                        path, current_cost = new_path, new_cost
                        improved = True

        end_time = time.time()
        return path, current_cost, end_time - start_time

    def calculate_total_cost(self, path):
        total_cost = 0
        for i in range(1, len(path)):
            total_cost += self.dist(path[i - 1], path[i])
        return total_cost



class TSPVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("TSP Visualizer with Backtracking")
        self.master.configure(bg="#f0f0f0")  # Background color of the main window
        
        # Create Canvas for cities and paths visualization
        self.canvas = tk.Canvas(self.master, width=900, height=400, bg="lightblue")
        self.canvas.grid(row=0, column=0, rowspan=5, padx=10, pady=10)
        
        # Bind click event on canvas
        self.canvas.bind("<Button-1>", self.canvas_click)

        # Data variables
        self.cities = []
        self.best_path = None
        self.city_count = 0  # To keep track of city number
        self.min_cost = float('inf')
        self.distances = []  # Distance matrix between cities

        # Listbox to show results
        self.result_listbox = tk.Listbox(self.master, width=70, height=20, bg="#f9f9f9", fg="black", font=("Arial", 10))
        self.result_listbox.grid(row=0, column=1, padx=10, pady=10, rowspan=4)

        # Control Frame for buttons
        self.control_frame = tk.Frame(self.master, bg="#f0f0f0")
        self.control_frame.grid(row=4, column=1, pady=10, sticky='w')

        # Buttons for controlling the program
        self.add_city_btn = tk.Button(self.control_frame, text="Thêm Thành Phố", command=self.add_city, bg="#4CAF50", fg="white", relief="raised", font=("Arial", 10))
        self.add_city_btn.grid(row=0, column=0, padx=5)

        self.start_btn = tk.Button(self.control_frame, text="Bắt Đầu (Backtracking)", command=self.start_backtracking, bg="#2196F3", fg="white", relief="raised", font=("Arial", 10))
        self.start_btn.grid(row=0, column=1, padx=5)

        self.reset_btn = tk.Button(self.control_frame, text="Reset", command=self.reset, bg="#f44336", fg="white", relief="raised", font=("Arial", 10))
        self.reset_btn.grid(row=0, column=2, padx=5)

        # Entry and Label for starting city
        self.start_city_label = tk.Label(self.control_frame, text="Chọn Thành Phố Bắt Đầu:", bg="#f0f0f0", font=("Arial", 10))
        self.start_city_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')

        self.start_city_entry = tk.Entry(self.control_frame, font=("Arial", 10), bg="#ffffff", fg="black")
        self.start_city_entry.grid(row=1, column=1, padx=5, pady=5)

        # Variable for drawing queue and lines
        self.draw_queue = deque()
        self.lines = []

        # Label for Distance Matrix
        self.distances_label = tk.Label(self.master, text="Ma Trận Khoảng Cách", font=("Arial", 12), bg="#f0f0f0")
        self.distances_label.grid(row=5, column=0, columnspan=1, pady=10)

        # Treeview to display distance matrix
        self.distance_tree = ttk.Treeview(self.master, columns=[""] + [str(i) for i in range(1, 41)], show="headings", height=6)
        self.distance_tree.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

        # Set column headings
        self.distance_tree.heading("#1", text="Thành Phố")
        for i in range(1, 41):  # Adjust number of columns as necessary
            self.distance_tree.heading(str(i), text=str(i))

        # Set column width
        self.distance_tree.column("#1", width=80, anchor="center")  # "City" column
        for i in range(1, 41):
            self.distance_tree.column(str(i), width=20, anchor="center")  # Distance columns

        # Create horizontal scrollbar for the Treeview
        self.scrollbar_x = ttk.Scrollbar(self.master, orient="horizontal", command=self.distance_tree.xview)
        self.scrollbar_x.grid(row=7, column=0, columnspan=2, sticky="ns")
        self.distance_tree.configure(xscrollcommand=self.scrollbar_x.set)
        self.distance_tree.configure(selectmode="extended")

        # Create vertical scrollbar for the Treeview
        self.scrollbar_y = ttk.Scrollbar(self.master, orient="vertical", command=self.distance_tree.yview)
        self.scrollbar_y.grid(row=6, column=2, sticky="ns")
        self.distance_tree.configure(yscrollcommand=self.scrollbar_y.set)

        # Update distance matrix when needed
        self.update_distance_matrix()

        # Add Checkbutton to skip simulation (direct result)
        self.skip_simulation_var = tk.BooleanVar()  # Variable to store checkbox state
        self.skip_simulation_checkbox = tk.Checkbutton(self.control_frame, text="Bỏ qua mô phỏng (Hiển thị kết quả trực tiếp)", 
                                                       variable=self.skip_simulation_var, font=("Arial", 10), bg="#f0f0f0")
        self.skip_simulation_checkbox.grid(row=2, column=0, columnspan=3, pady=5)

        self.speed_var = tk.StringVar(value="Bình Thường")  # Default value is "Bình Thường"
        self.speed_option = tk.OptionMenu(self.control_frame, self.speed_var, "Bình Thường", "X2")
        self.speed_option.grid(row=3, column=0, padx=5, pady=5)

        # Thêm các biến đếm
        self.num_iterations = 0  # Đếm số bước lặp
        self.num_pruned_branches = 0  # Đếm số nhánh bị cắt
        self.num_backtracks = 0  # Đếm số lần backtrack
        # Text widget để hiển thị đường đi tốt nhất hiện tại
        self.best_path_text = tk.Text(self.master, height=5, width=50, bg="#37426F", fg="black", font=("Arial", 10))
        self.best_path_text.grid(row=5, column=1, padx=10, pady=5, sticky="nsew")

        self.results = {}
        self.min_dis = 0

        self.performance_canvas = tk.Canvas(self.master, width=900, height=200, bg="white")
        self.performance_canvas.grid(row=8, column=0, columnspan=2, padx=10, pady=10)
        # Listbox để hiển thị thống kê các thuật toán (Full chiều rộng phải màn hình)
        self.algo_result_listbox = tk.Listbox(self.master, width=100, height=15, bg="#f0f0f0", fg="black", font=("Arial", 10))
        self.algo_result_listbox.grid(row=6, column=2, columnspan=1, rowspan=3, padx=10, pady=10, sticky="nsew")

    def draw_performance_chart(self, results):
        self.performance_canvas.delete("all")
        bar_width = 100
        max_cost = max(cost for cost, _ in results.values())

        for i, (algo, (cost, _)) in enumerate(results.items()):
            x0 = i * bar_width + 50
            y0 = 200 - (cost / max_cost * 180)
            x1 = x0 + bar_width - 10
            y1 = 200
            self.performance_canvas.create_rectangle(x0, y0, x1, y1, fill="blue")
            self.performance_canvas.create_text((x0 + x1) // 2, y0 - 10, text=f"{cost}", font=("Arial", 10))
            self.performance_canvas.create_text((x0 + x1) // 2, y0 + 50, text=algo, font=("Arial", 10), fill="white")
    def run_algorithm(self):
        start_city = int(self.start_city_entry.get()) - 1
        algorithms = TSPAlgorithms(self.cities)


        self.algo_result_listbox.delete(0, tk.END)  # Clear previous algorithm results

        path, cost, exec_time_greedy = algorithms.greedy(start_city)
        #Greedy
        self.results["Greedy"] = (cost, exec_time_greedy)
        self.algo_result_listbox.insert(tk.END, f"Greedy Algorithm:")
        self.algo_result_listbox.insert(tk.END, f"  Path: {' -> '.join(map(str, [p + 1 for p in path]))}")
        self.algo_result_listbox.insert(tk.END, f"  Cost: {cost}")
        self.algo_result_listbox.insert(tk.END, f"  Time: {exec_time_greedy:.2f} seconds")
        path, cost, exec_time_hill_climbing = algorithms.hill_climbing(start_city)
        #Hill Climbing
        self.results["Hill Climbing"] = (cost, exec_time_hill_climbing)
        self.algo_result_listbox.insert(tk.END, f"Hill Climbing:")
        self.algo_result_listbox.insert(tk.END, f"  Path: {' -> '.join(map(str, [p + 1 for p in path]))}")
        self.algo_result_listbox.insert(tk.END, f"  Cost: {cost}")
        self.algo_result_listbox.insert(tk.END, f"  Time: {exec_time_hill_climbing:.2f} seconds")
        
         # Minimax
        cost, path, exec_time_minimax = algorithms.solve_minimax(start_city)
        self.results["Minimax"] = (cost, exec_time_minimax)  # Minimax execution time might not be meaningful
        self.algo_result_listbox.insert(tk.END, f"Minimax Algorithm:")
        self.algo_result_listbox.insert(tk.END, f"  Path: {' -> '.join(map(str, [p + 1 for p in path]))}")
        self.algo_result_listbox.insert(tk.END, f"  Cost: {cost}")
        self.algo_result_listbox.insert(tk.END, f"  Time: {exec_time_minimax:.2f} seconds")  # Use exec_time if you measure it

        # Genetic Algorithm (GA)
        path, cost, exec_time_genetic = algorithms.genetic_algorithm(population_size=100, generations=500, mutation_rate=0.05, start_city=start_city)
        self.results["Genetic Algorithm"] = (cost, exec_time_genetic)  # Execution time optional for GA
        self.algo_result_listbox.insert(tk.END, f"Genetic Algorithm:")
        self.algo_result_listbox.insert(tk.END, f"  Path: {' -> '.join(map(str, [p + 1 for p in path]))}")
        self.algo_result_listbox.insert(tk.END, f"  Cost: {cost}")
        self.algo_result_listbox.insert(tk.END, f"  Time: {exec_time_genetic:.2f} seconds")  # Use exec_time if calculated

        # Simulated Annealing (SA)
        path, cost, exec_time_simulated = algorithms.simulated_annealing(start_city, initial_temp=1000, cooling_rate=0.99, stopping_temp=1)
        self.results["Simulated Annealing"] = (cost, exec_time_simulated)  # Execution time optional for SA
        self.algo_result_listbox.insert(tk.END, f"Simulated Annealing:")
        self.algo_result_listbox.insert(tk.END, f"  Path: {' -> '.join(map(str, [p + 1 for p in path]))}")
        self.algo_result_listbox.insert(tk.END, f"  Cost: {cost}")
        self.algo_result_listbox.insert(tk.END, f"  Time: {exec_time_simulated:.2f} seconds")  # Use exec_time if measured

        if not self.results:
            self.algo_result_listbox.insert(tk.END, "No results to display.")
        else:
            self.draw_performance_chart(self.results)
    def canvas_click(self, event):
        # Get the coordinates of the click
        city_x, city_y = event.x, event.y

        # Check if the city is at least 30 units away from all other cities
        valid_position = True
        for city in self.cities:
            x1, y1 = city
            distance = math.sqrt((city_x - x1) ** 2 + (city_y - y1) ** 2)
            if distance < 40:  # If distance is less than 40, it's an invalid position
                valid_position = False
                break

        if valid_position:
            # Add city coordinates to the list
            self.cities.append((city_x, city_y))
            self.city_count += 1  # Increment city count for the next city

            # Draw the city as a circle
            city_id = self.canvas.create_oval(city_x-5, city_y-5, city_x+5, city_y+5, fill="blue")

            # Add city number (STT) on top of the city
            self.canvas.create_text(city_x, city_y - 15, text=str(self.city_count), font=("Arial", 10, "bold"), fill="black")

            # Recalculate distances and update the matrix
            self.calculate_distances()
            self.update_distance_matrix()
        else:
            print("Invalid position: too close to an existing city.")

    def update_distance_matrix(self):
            # Cập nhật ma trận khoảng cách trong bảng

            for row in self.distance_tree.get_children():
                self.distance_tree.delete(row)
            
            for i in range(len(self.distances)):
                self.distance_tree.insert("", "end", values=[str(i+1)] + self.distances[i])

    def calculate_distances(self):
        # Tính toán ma trận khoảng cách giữa các thành phố
        n = len(self.cities)
        self.distances = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance = ((self.cities[i][0] - self.cities[j][0]) ** 2 + (self.cities[i][1] - self.cities[j][1]) ** 2) ** 0.5
                    self.distances[i][j] = math.ceil(distance)
                    self.min_dis = min(self.min_dis, self.distances[i][j])
    
    def add_city(self):
        # Try to find a valid position for the new city
        max_tries = 100  # Limit number of attempts to avoid infinite loop
        for _ in range(max_tries):
            # Generate a random position for the new city
            city_x = random.randint(50, 550)
            city_y = random.randint(50, 350)

            # Check if the city is at least 30 units away from all other cities
            valid_position = True
            for city in self.cities:
                x1, y1 = city
                distance = math.sqrt((city_x - x1) ** 2 + (city_y - y1) ** 2)
                if distance < 40:  # If distance is less than 30, it's an invalid position
                    valid_position = False
                    break

            if valid_position:
                # Add city coordinates to the list
                self.cities.append((city_x, city_y))
                self.city_count += 1  # Increment city count for the next city

                # Draw the city as a circle
                city_id = self.canvas.create_oval(city_x-5, city_y-5, city_x+5, city_y+5, fill="blue")

                # Add city number (STT) on top of the city
                self.canvas.create_text(city_x, city_y - 15, text=str(self.city_count), font=("Arial", 10, "bold"), fill="black")
                break  # Exit the loop if a valid position is found
        else:
            print("Could not find a valid position after several tries.")

        self.calculate_distances()
        self.update_distance_matrix()

    def start_backtracking(self):
        """Khởi động giải thuật TSP sử dụng Backtracking"""
        if len(self.cities) < 2:
            self.result_listbox.insert(tk.END, "Cần ít nhất 2 thành phố để bắt đầu.")
            return
        
        # Lấy thành phố khởi đầu từ ô input
        try:
            start_city = int(self.start_city_entry.get()) - 1
            if start_city < 0 or start_city >= len(self.cities):
                raise ValueError("Thành phố khởi đầu không hợp lệ.")
        except ValueError:
            self.result_listbox.insert(tk.END, "Vui lòng nhập một thành phố hợp lệ.")
            return

        self.result_listbox.delete(0, tk.END)
        self.run_algorithm()
        self.min_cost = float('inf')
        self.best_path = None
        self.path_lines = []
        visited = [False] * len(self.cities)
        path = [start_city]
        visited[start_city] = True
        self.best_path_holder = []
        self.best_path_lineid = []
        # Đo thời gian bắt đầu
        start_time = time.time()

        # Bắt đầu giải thuật TSP
        self.solve_tsp_with_backtracking(start_city, visited, path, 0)

        # Đo thời gian kết thúc
        end_time = time.time()
        self.time = end_time - start_time

        # Sau khi tìm xong, xử lý hàng đợi vẽ
        self.process_draw_queue()



    def solve_tsp_with_backtracking(self, current_city, visited, path, current_cost):
        """Backtracking để giải bài toán TSP với heuristic kết hợp"""
        if len(path) == len(self.cities):
            total_cost = current_cost + self.dist(current_city, path[0])
            self.draw_queue.append(("fullpath", -1, -1, path[:], total_cost))
            if total_cost < self.min_cost:
                self.min_cost = total_cost
                self.best_path = path[:]
                self.best_path.append(path[0])
                passing_path = self.best_path[:]
                passing_min_cost = self.min_cost
                self.draw_queue.append(("updatefullpath", -1, -1, passing_path, passing_min_cost))
            return

        # Đếm số bước lặp
        self.num_iterations += 1

        # Sắp xếp các thành phố chưa ghé thăm theo Nearest Neighbor Heuristic
        remaining_cities = [
            city for city in range(len(self.cities)) if not visited[city]
        ]
        sorted_cities = sorted(
            remaining_cities, key=lambda city: self.dist(current_city, city)
        )

        for next_city in sorted_cities:
            lower_bound = float('inf')
            if self.min_cost != float('inf'):
                lower_bound = self.estimate_lower_bound(path, visited)
            #print(f"current_city = {current_city} | current_cost = {current_cost} | current_cost + lower_bound = {current_cost + lower_bound} | self.min_cost = {self.min_cost}")
            if lower_bound != float('inf') and current_cost + lower_bound > self.min_cost:
                 # Cắt nhánh
                self.num_pruned_branches += 1
                path.append(next_city)
                self.draw_queue.append(("draw", path[-2], path[-1], None, None))
                self.draw_queue.append(("remove", path[-2], path[-1], None, None))
                path.pop()
                continue  # Cắt nhánh nếu chi phí hiện tại cộng với lower bound đã vượt quá min_cost

            # Đánh dấu thành phố đã ghé thăm
            visited[next_city] = True
            path.append(next_city)
            self.draw_queue.append(("draw", path[-2], path[-1], None, None))

            # Đệ quy tiếp tục tìm kiếm
            self.solve_tsp_with_backtracking(next_city, visited, path, current_cost + self.dist(current_city, next_city))

            # Sau khi backtrack, lưu thao tác xóa đường vào hàng đợi
            self.draw_queue.append(("remove", path[-2], path[-1], None, None))
            path.pop()
            visited[next_city] = False
            self.num_backtracks += 1
    
    def estimate_lower_bound(self, path, visited):
        """Lower Bound dùng MST để ước lượng chi phí tối thiểu (Sử dụng thuật toán Prim để tối ưu hơn)"""
        remaining_cities = [i for i in range(len(self.cities)) if not visited[i]]
        
        if not remaining_cities:
            return 0  # Không còn thành phố nào chưa ghé thăm

        # Dùng thuật toán Prim để tính MST cho các thành phố còn lại
        import heapq # Sử dụng PriorityQueue (Hay còn gọi là min_heap trong Python)

        mst_cost = 0
        min_heap = []
        # Khởi tạo heap với các cạnh từ thành phố đầu tiên
        start_city = remaining_cities[0]
        for city in remaining_cities[1:]:
            heapq.heappush(min_heap, (self.dist(start_city, city), start_city, city))

        # Dùng heap để xây dựng MST
        visited_cities = {start_city}
        while min_heap:
            dist, city1, city2 = heapq.heappop(min_heap) # lấy ra phần tử
            if city2 not in visited_cities:
                visited_cities.add(city2)
                mst_cost += dist
                # Thêm các cạnh mới vào heap
                for city in remaining_cities:
                    if city not in visited_cities:
                        heapq.heappush(min_heap, (self.dist(city2, city), city2, city))

        return mst_cost
    
    def process_draw_queue(self):
        """Xử lý các thao tác vẽ trong hàng đợi"""
        if not self.skip_simulation_var.get():
            if self.draw_queue:
                action, city1, city2, path, cost = self.draw_queue.popleft()
                if action == "fullpath":
                    totalcost = cost
                    current_path = path[:]
                    current_path.append(current_path[0])
                    fullpath_str = " -> ".join(str(node + 1) for node in current_path)
                    self.result_listbox.insert(tk.END, "==========================")
                    self.result_listbox.insert(tk.END, f"Hoàn thành đường: {fullpath_str}")
                    self.result_listbox.insert(tk.END, f"Chi Phí = {totalcost}")
                    self.result_listbox.insert(tk.END, "==========================")
                elif action == "updatefullpath":
                    my_best_path = path
                    min_cost = cost
                    bestfullpath_str = " -> ".join(str(node + 1) for node in my_best_path)

                    # Xóa nội dung cũ trong Text widget
                    self.best_path_text.delete(1.0, tk.END)

                    # Chèn thông tin mới với định dạng đẹp
                    self.best_path_text.insert(tk.END, "Đường tốt nhất hiện tại:\n", "title")
                    self.best_path_text.insert(tk.END, f"{bestfullpath_str}\n", "path")
                    self.best_path_text.insert(tk.END, f"Chi phí: {min_cost}\n", "cost")

                    # Cấu hình màu sắc và font chữ cho từng phần
                    self.best_path_text.tag_configure("title", foreground="#B6020E", font=("Arial", 14, "bold"))
                    self.best_path_text.tag_configure("path", foreground="#EE8980", font=("Arial", 12, "bold"))
                    self.best_path_text.tag_configure("cost", foreground="#A9D0D5", font=("Arial", 12, "bold"))
                    #self.clear_best_path_holder()
                    #self.draw_placeholder_best_path(my_best_path)
                elif action == "draw":
                    x1, y1 = self.cities[city1]
                    x2, y2 = self.cities[city2]
                    line_id = self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2)
                    self.lines.append(line_id)
                elif action == "remove" and self.lines:
                    self.canvas.delete(self.lines.pop())

                # Tiếp tục xử lý thao tác tiếp theo sau một khoảng thời gian
                speed = self.speed_var.get()
                delay = 300 if speed == "Bình Thường" else 150
                self.master.after(delay, self.process_draw_queue)
                return
        self.draw_best_path()
    def draw_placeholder_best_path(self, mypath):
        """Vẽ đường đi tốt nhất hiện tại dưới dạng màu xám."""
        # Xóa dữ liệu cũ
        self.best_path_holder = []  # Danh sách lưu thông tin đường
        self.best_path_lineid = []  # Danh sách lưu ID các đường đã vẽ

        for i in range(len(mypath) - 1):
            x1, y1 = self.cities[mypath[i]]
            x2, y2 = self.cities[mypath[i + 1]]
            self.best_path_holder.append((x1, y1, x2, y2))

        # Vẽ tất cả các đoạn đường màu xám
        for segment in self.best_path_holder:
            x1, y1, x2, y2 = segment
            line_id = self.canvas.create_line(x1, y1, x2, y2, fill="grey", width=2)
            self.best_path_lineid.append(line_id)

    def clear_best_path_holder(self):
        """Xóa tất cả các đoạn đường tốt nhất hiện tại."""
        for line_id in self.best_path_lineid:
            self.canvas.delete(line_id)
        self.best_path_lineid = []

    def draw_best_path(self):
        """Vẽ đường đi tối ưu với màu đỏ trước, sau đó chuyển sang màu xanh lá cho các đoạn đã vẽ"""
        if not self.best_path:
            return

        # Tạo danh sách các đoạn đường cần vẽ
        for i in range(len(self.best_path) - 1):
            x1, y1 = self.cities[self.best_path[i]]
            x2, y2 = self.cities[self.best_path[i+1]]
            self.path_lines.append((x1, y1, x2, y2))

        # Vẽ tất cả các đoạn đường với màu đỏ trước
        def draw_all_red(i):
            if i < len(self.path_lines):
                x1, y1 =  self.path_lines[i][0],  self.path_lines[i][1]
                x2, y2 =  self.path_lines[i][2],  self.path_lines[i][3]
                line_id = self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2)
                self.lines.append(line_id)
                self.master.after(500, draw_all_red, i + 1)

        # Sau khi vẽ tất cả các đoạn với màu đỏ, chuyển màu sang xanh
        def change_to_green(i):
            if i < len( self.path_lines):
                x1, y1 =  self.path_lines[i][0],  self.path_lines[i][1]
                x2, y2 =  self.path_lines[i][2],  self.path_lines[i][3]
                self.canvas.create_line(x1, y1, x2, y2, fill="green", width=2)
                self.canvas.delete(self.lines[i])  # Xóa đoạn đường đỏ cũ
                self.master.after(500, change_to_green, i + 1)

        # Vẽ tất cả các đoạn đường đỏ
        draw_all_red(0)

        bestfullpath_str = " -> ".join(str(node + 1) for node in self.best_path)

        # Xóa nội dung cũ trong Text widget
        self.best_path_text.delete(1.0, tk.END)

        # Chèn thông tin mới với định dạng đẹp
        self.best_path_text.insert(tk.END, "Đường tốt nhất hiện tại:\n", "title")
        self.best_path_text.insert(tk.END, f"{bestfullpath_str}\n", "path")
        self.best_path_text.insert(tk.END, f"Chi phí: {self.min_cost}\n", "cost")
        # Cấu hình màu sắc và font chữ cho từng phần
        self.best_path_text.tag_configure("title", foreground="#B6020E", font=("Arial", 14, "bold"))
        self.best_path_text.tag_configure("path", foreground="#EE8980", font=("Arial", 12, "bold"))
        self.best_path_text.tag_configure("cost", foreground="#A9D0D5", font=("Arial", 12, "bold"))

        # Sau khi tất cả đoạn đường đỏ được vẽ, chuyển thành màu xanh
        time = 500 * len( self.path_lines)
        self.master.after(time, lambda: change_to_green(0))

        # Hiển thị kết quả trong Listbox
        if self.best_path:
            path_str = " -> ".join(str(node + 1) for node in self.best_path)
            cost_str = f"Chi Phí Tối Thiểu: {self.min_cost}"
            time_str = f"Thời gian chạy: {self.time:.11f} giây"
            self.result_listbox.insert(tk.END, f"Số bước lặp: {self.num_iterations}")
            self.result_listbox.insert(tk.END, f"Số nhánh bị cắt: {self.num_pruned_branches}")
            self.result_listbox.insert(tk.END, f"Số lần backtrack: {self.num_backtracks}")
            self.result_listbox.insert(tk.END, f"Đường đi tối ưu với Heuristic:")
            self.result_listbox.insert(tk.END, f" {path_str}")
            self.result_listbox.insert(tk.END, cost_str)
            self.result_listbox.insert(tk.END, time_str)  # Hiển thị thời gian chạy

    def dist(self, city1, city2):
        """Tính khoảng cách Euclidean giữa 2 thành phố"""
        x1, y1 = self.cities[city1]
        x2, y2 = self.cities[city2]
        return math.ceil(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    def reset(self):
        # Reset các thành phố và ma trận khoảng cách
        self.cities = []
        self.distances = []
        self.draw_queue = deque()
        self.city_count = 0
        self.best_path = None
        self.path_lines = []
        self.min_cost = float('inf')
        self.num_iterations = 0  # Đếm số bước lặp
        self.num_pruned_branches = 0  # Đếm số nhánh bị cắt
        self.num_backtracks = 0  # Đếm số lần backtrack
        # Xóa các thành phố trên canvas và reset bảng
        self.canvas.delete("all")
        self.result_listbox.delete(0, tk.END)
        self.update_distance_matrix()

# Thiết lập giao diện chính
root = tk.Tk()
tsp_visualizer = TSPVisualizer(root)
root.mainloop()
