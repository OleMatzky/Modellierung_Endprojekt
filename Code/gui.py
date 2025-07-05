import os
import sys
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from itertools import chain
import numpy as np
from numpy.random import Generator, PCG64
from typing import Iterator
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from source.grid import Grid 
from source.search_base import SearchAlgo
from source.astar import AStar
from source.dijkstra import Dijkstra
from source.greedy import GreedyBFS

# ---------------------------------------- CONSTANTS ----------------------------------------
CELL_COLORS = {
    0: 'white',      # Free space
    1: 'black',      # Wall
    2: 'green',      # Start position
    3: 'red',        # Goal position
    4: 'yellow',     # Open nodes
    5: 'lightblue',  # Current path
    6: 'blue'        # Final path
}
MAZE_COLORMAP = ListedColormap([CELL_COLORS[i] for i in range(len(CELL_COLORS))])

# ---------------------------------------- MAIN GUI CLASS ----------------------------------------
class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Intelligent Pathfinding: Comparative Simulation of Search Strategies in Generated Maze Environments")
        self.protocol("WM_DELETE_WINDOW", self._on_application_close)

        # Available pathfinding algorithms
        self.available_algorithms: list[SearchAlgo] = [AStar(), Dijkstra(), GreedyBFS()]

        # -------------------- UI SETUP --------------------
        self._setup_user_interface()
        self._setup_matplotlib_canvas()
        
        # -------------------- STATE VARIABLES --------------------
        self.maze_generator = Grid()
        self.current_animation: FuncAnimation | None = None
        self.frame_iterator: Iterator | None = None
        self.delay_callback_id = None
        self.animation_mode = None
        self.current_algorithm_index: int = -1
        self.active_algorithms = []
        self.results_table = None

    # -------------------- UI CONSTRUCTION --------------------
    def _setup_user_interface(self):
        """Create and configure all UI elements."""
        main_container = tk.Frame(self)
        main_container.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        self._create_maze_parameter_controls(main_container)
        self._create_algorithm_selection_controls(main_container)
        self._create_control_buttons(main_container)

    def _create_maze_parameter_controls(self, parent_frame):
        """Create maze generation parameter controls."""
        maze_config_frame = ttk.LabelFrame(parent_frame, text="Maze Parameters")
        maze_config_frame.pack()

        # Dimension sliders
        self.width_slider = tk.Scale(
            maze_config_frame, from_=30, to=300, label="Width", 
            orient="horizontal", resolution=10
        )
        self.height_slider = tk.Scale(
            maze_config_frame, from_=20, to=200, label="Height",
            orient="horizontal", resolution=10
        )
        self.extra_openings_slider = tk.Scale(
            maze_config_frame, from_=0, to=10, label="Extra openings (%)", 
            orient="horizontal"
        )

        # Set default values
        self.width_slider.set(100)
        self.height_slider.set(60)
        self.extra_openings_slider.set(5)

        # Layout sliders
        for row_index, slider in enumerate([self.width_slider, self.height_slider, self.extra_openings_slider]):
            slider.grid(row=row_index, column=0, padx=5, pady=4, sticky="we")

        # Animation toggle checkbox
        self.animate_maze_generation_var = tk.BooleanVar(value=True)
        animation_checkbox = ttk.Checkbutton(
            maze_config_frame, text="Animate Generation", 
            variable=self.animate_maze_generation_var
        )
        animation_checkbox.grid(row=3, column=0, padx=5, pady=4, sticky="w")

        # Random seed controls
        self.use_random_seed_var = tk.BooleanVar(value=True)
        random_seed_checkbox = ttk.Checkbutton(
            maze_config_frame, text="Random Seed", 
            variable=self.use_random_seed_var
        )
        random_seed_checkbox.grid(row=4, column=0, padx=5, pady=4, sticky="we")

        self.seed_input_frame = ttk.Frame(maze_config_frame)
        self.seed_input_frame.grid(row=5, column=0, padx=5, pady=4, sticky="we")
        self.seed_input_frame.grid_remove()
        
        seed_label = tk.Label(self.seed_input_frame, text="Seed:")
        seed_label.pack(side="left")
        
        self.seed_value_var = tk.StringVar(value="42")
        seed_input_spinbox = tk.Spinbox(
            self.seed_input_frame, from_=0, to=1000, width=10, increment=1, 
            state="readonly", justify="right", textvariable=self.seed_value_var, 
            repeatinterval=50
        )
        seed_input_spinbox.pack(side="left", pady=4)

        def toggle_seed_input_visibility():
            if self.use_random_seed_var.get():
                self.seed_input_frame.grid_remove()
            else:
                self.seed_input_frame.grid()
        
        random_seed_checkbox.config(command=toggle_seed_input_visibility)

    def _create_algorithm_selection_controls(self, parent_frame):
        """Create algorithm selection checkboxes."""
        algorithm_selection_frame = ttk.Labelframe(parent_frame, text="Algorithms")
        algorithm_selection_frame.pack(pady=10, anchor="nw")
        
        self.algorithm_enabled_vars = [tk.BooleanVar(value=True) for _ in self.available_algorithms]
        
        for algo_index, algorithm in enumerate(self.available_algorithms):
            algorithm_checkbox = ttk.Checkbutton(
                algorithm_selection_frame, 
                text=algorithm.name, 
                command=lambda i=algo_index: self._on_algorithm_selection_changed(i), 
                variable=self.algorithm_enabled_vars[algo_index]
            )
            algorithm_checkbox.grid(row=algo_index, column=0, padx=5, pady=4, sticky="w")

    def _create_control_buttons(self, parent_frame):
        """Create control buttons and settings."""
        control_panel_frame = ttk.Labelframe(parent_frame, text="Controls")
        control_panel_frame.pack(anchor="nw")

        # Animation speed control
        self.animation_delay_slider = tk.Scale(
            control_panel_frame, from_=10, to=1000, label="Delay (ms)", 
            resolution=10, orient="horizontal", 
            command=lambda _: self._schedule_animation_speed_update()
        )
        self.animation_delay_slider.grid(row=0, column=0, padx=5, pady=4, sticky="we")
        self.animation_delay_slider.set(50)

        # Action buttons
        ttk.Button(
            control_panel_frame, text="Start Animation",
            command=self.start_pathfinding_animation
        ).grid(row=1, column=0, padx=5, pady=4, sticky="w")
        
        ttk.Button(
            control_panel_frame, text="Save Animation (MP4)",
            command=self.export_animation_to_mp4
        ).grid(row=2, column=0, padx=5, pady=4, sticky="w")

    def _setup_matplotlib_canvas(self):
        """Initialize matplotlib figure and canvas."""
        self.visualization_figure: plt.Figure = plt.Figure(figsize=(10, 6))
        self.title_display: Text = self.visualization_figure.text(
            0.5, 0.95, "", ha='center', fontsize=14, weight='bold'
        )
        
        # Main maze visualization area
        self.maze_display_axes: Axes = self.visualization_figure.add_axes([0, 0, 1, 0.92])
        self.maze_display_axes.axis('off')
        
        # Results table area
        self.results_table_axes: Axes = self.visualization_figure.add_axes([0.35, 0.17, 0.3, 0.25])
        self.results_table_axes.axis('off')

        self.matplotlib_canvas = FigureCanvasTkAgg(self.visualization_figure, master=self)
        self.matplotlib_canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

    # -------------------- EVENT HANDLERS --------------------
    def _on_algorithm_selection_changed(self, algorithm_index: int):
        """Ensure at least one algorithm is always selected."""
        if not any(var.get() for var in self.algorithm_enabled_vars):
            self.algorithm_enabled_vars[algorithm_index].set(True)

    def _schedule_animation_speed_update(self):
        """Schedule animation speed update to prevent multiple rapid calls."""
        if self.delay_callback_id:
            self.after_cancel(self.delay_callback_id)

        if self.animation_mode == "maze":
            self.delay_callback_id = self.after(
                500, lambda: self._apply_animation_speed_change(
                    self._update_maze_frame, use_blitting=True
                )
            )
        elif self.animation_mode == "algorithm":
            self.delay_callback_id = self.after(
                500, lambda: self._apply_animation_speed_change(
                    self._update_algorithm_frame, use_blitting=False
                )
            )

    def _apply_animation_speed_change(self, update_function: callable, use_blitting: bool):
        """Apply new animation speed to current animation."""
        if not (self.current_animation and self.frame_iterator):
            return

        self.current_animation.event_source.stop()
        self.current_animation = None

        new_delay = self.animation_delay_slider.get()

        self.current_animation = FuncAnimation(
            self.visualization_figure, update_function,
            frames=self.frame_iterator, interval=new_delay, 
            cache_frame_data=False, blit=use_blitting, repeat=False
        )
        self.matplotlib_canvas.draw_idle()

    # -------------------- MAZE GENERATION --------------------
    def get_maze_generation_frame_iterator(self, should_animate: bool = True) -> Iterator[np.ndarray]:
        """Generate maze frames for visualization."""
        maze_width = self.width_slider.get()
        maze_height = self.height_slider.get()
        extra_openings_percent = self.extra_openings_slider.get()

        seed_value = int(self.seed_value_var.get()) if not self.use_random_seed_var.get() else None
        random_generator = Generator(PCG64(seed_value))
        
        return self.maze_generator.generate_maze_grid(
            maze_width, maze_height, random_generator, 
            extra_openings=extra_openings_percent, animate=should_animate
        )

    def _update_maze_frame(self, maze_data: np.ndarray):
        """Update maze visualization frame during generation."""
        if maze_data.shape == (1,):
            # Sentinel value indicates generation completion
            self.current_animation.event_source.stop()
            self.after(500, self.start_all_pathfinding_algorithms)
            return []

        # Measure Rendering FPS
        if not hasattr(self, "_t0"):
            self._t0 = time.perf_counter()
            self.live_fps_list = []
        if not hasattr(self, "_frames_done"):
            self._frames_done = 0
        self._frames_done += 1

        if self._frames_done % 50 == 0 and len(self.live_fps_list) < 6:
            now = time.perf_counter()
            self.live_fps = 50 / (now - self._t0 - 50 * (self.animation_delay_slider.get() / 1000))
            self.live_fps_list.append(self.live_fps)
            self._t0 = now

        self.maze_image_display.set_data(maze_data)
        return (self.maze_image_display,)

    # -------------------- ANIMATION CONTROL --------------------
    def start_pathfinding_animation(self):
        """Start the complete pathfinding animation sequence."""
        self.animation_mode = "maze"
        animation_delay = self.animation_delay_slider.get()
        should_animate_maze = self.animate_maze_generation_var.get()
        
        if should_animate_maze:
            self.title_display.set_text("Maze Generation")

        self.frame_iterator = self.get_maze_generation_frame_iterator(should_animate_maze)

        # Reset visualization areas
        self.maze_display_axes.clear()
        self.results_table_axes.clear()
        self.maze_display_axes.axis('off')
        self.results_table_axes.axis('off')

        if should_animate_maze:
            self.maze_display_axes.set_position([0, 0, 1, 0.92])
        else:
            self.maze_display_axes.set_position([0, 0.2, 1, 0.7])

        self.maze_image_display = self.maze_display_axes.imshow(
            next(self.frame_iterator), cmap=MAZE_COLORMAP,
            vmin=0, vmax=len(CELL_COLORS)-1, aspect='auto'
        )

        if self.current_animation and self.current_animation.event_source is not None:
            self.current_animation.event_source.stop()

        if should_animate_maze:
            self.current_animation = FuncAnimation(
                self.visualization_figure, self._update_maze_frame,
                frames=self.frame_iterator, cache_frame_data=False,
                interval=animation_delay, blit=True, repeat=False
            )
        else:
            self.start_all_pathfinding_algorithms()
        
        self.matplotlib_canvas.draw()

    def start_all_pathfinding_algorithms(self):
        """Initialize and start pathfinding algorithm visualization."""
        self.animation_mode = "algorithm"
        self.active_algorithms = [
            algorithm for algorithm, is_enabled in zip(self.available_algorithms, self.algorithm_enabled_vars) 
            if is_enabled.get()
        ]
        
        if not self.active_algorithms:
            return
        
        # Create results table
        algorithm_names = [algo.name for algo in self.active_algorithms]
        self.results_table = self.results_table_axes.table(
            cellText=[["-", "-", "-"] for _ in algorithm_names],
            rowLabels=algorithm_names,
            colLabels=["# Nodes", "Length", "Time[ms]"],
            loc='bottom'
        )
        self.results_table.scale(1, 1.5)
        self.maze_display_axes.set_position([0, 0.2, 1, 0.7])

        self.current_algorithm_index = -1
        self._start_next_algorithm()

    def _start_next_algorithm(self):
        """Start the next algorithm in the sequence."""
        self.current_algorithm_index += 1

        if self.current_algorithm_index >= len(self.active_algorithms):
            return  # All algorithms completed

        current_algorithm: SearchAlgo = self.active_algorithms[self.current_algorithm_index]
        
        # Set appropriate title
        if current_algorithm.name != "Greedy BFS":
            self.title_display.set_text(f"Running {current_algorithm.name} Algorithm")
        else:
            self.title_display.set_text(f"Running Greedy Best-First Search Algorithm")

        self.frame_iterator = current_algorithm.run(
            self.maze_generator.grid.copy(),
            self.maze_generator.start,
            self.maze_generator.goal
        )

        # Start new animation
        if self.current_animation and self.current_animation.event_source is not None: 
            self.current_animation.event_source.stop()

        animation_delay = self.animation_delay_slider.get()
        self.current_animation = FuncAnimation(
            self.visualization_figure, self._update_algorithm_frame,
            frames=self.frame_iterator, interval=animation_delay,
            cache_frame_data=False, blit=False, repeat=False
        )
        self.matplotlib_canvas.draw_idle()

    def _update_algorithm_frame(self, frame_data: tuple[np.ndarray, tuple[int, int, int]]):
        """Update algorithm visualization frame."""
        maze_grid, performance_metrics = frame_data

        if maze_grid.shape == (1,):
            # Sentinel value indicates algorithm completion
            self.current_animation.event_source.stop()
            self.after(1000, self._start_next_algorithm)
            return []

        self.maze_image_display.set_data(maze_grid)

        nodes_visited, path_length, execution_time_ms = performance_metrics
        table_row_index = self.current_algorithm_index + 1
        
        # Update table cells with new metrics
        if not np.isnan(nodes_visited):
            nodes_cell = self.results_table[(table_row_index, 0)].get_text()
            nodes_cell.set_text(str(nodes_visited))
        
        if not np.isnan(path_length):
            length_cell = self.results_table[(table_row_index, 1)].get_text()
            length_cell.set_text(str(path_length))
        
        if not np.isnan(execution_time_ms):
            time_cell = self.results_table[(table_row_index, 2)].get_text()
            time_cell.set_text(str(execution_time_ms))

        return (self.maze_image_display, self.results_table,)

    # -------------------- VIDEO EXPORT --------------------
    def export_animation_to_mp4(self):
        """Export complete animation sequence to MP4 file."""
        should_include_maze_generation = self.animate_maze_generation_var.get()
        
        if should_include_maze_generation:
            user_choice = messagebox.askyesnocancel(
                "Include Maze Generation in Animation?",
                "Including the maze generation process will make the animation significantly longer. "
                "Do you want to proceed with the full animation?",
                default='no', icon=messagebox.WARNING
            )
            
            if user_choice is None:
                return
            should_include_maze_generation = user_choice

        script_directory = os.path.dirname(os.path.abspath(__file__))
        output_filename = filedialog.asksaveasfilename(
            defaultextension=".mp4", initialdir=script_directory,
            filetypes=[("MP4 files", "*.mp4")]
        )
        
        if not output_filename:
            return
        
        maze_frame_iterator = self.get_maze_generation_frame_iterator(should_include_maze_generation)
        
        def wrap_maze_frames(frame_generator: Iterator[np.ndarray]) -> Iterator[dict]:
            """Convert maze frames to standardized format."""
            for maze_grid in frame_generator:
                if maze_grid.shape == (1,):
                    continue  # Skip sentinel values
                yield {
                    "image": maze_grid,
                    "algorithm_index": -1,
                    "nodes_visited": np.nan,
                    "path_length": np.nan,
                    "execution_time": np.nan
                }
                
        maze_frame_sequence = list(wrap_maze_frames(maze_frame_iterator))

        def wrap_algorithm_frames(frame_generator: Iterator[tuple[np.ndarray, tuple[int, int, int]]], 
                                algorithm_index: int) -> Iterator[dict]:
            """Convert algorithm frames to standardized format."""
            for maze_grid, (nodes, path_len, time_ms) in frame_generator:
                if maze_grid.shape == (1,):
                    continue  # Skip sentinel values
                yield {
                    "image": maze_grid,
                    "algorithm_index": algorithm_index,
                    "nodes_visited": nodes,
                    "path_length": path_len,
                    "execution_time": time_ms
                }
                
        base_maze_grid = self.maze_generator.grid.copy()
        start_position, goal_position = self.maze_generator.get_start_goal()
        frame_sequence_list = [maze_frame_sequence]

        selected_algorithms: list[SearchAlgo] = [
            algorithm for algorithm, is_enabled in zip(self.available_algorithms, self.algorithm_enabled_vars) 
            if is_enabled.get()
        ]

        for algo_index, algorithm in enumerate(selected_algorithms):
            algorithm_frames = wrap_algorithm_frames(
                algorithm.run(base_maze_grid.copy(), start_position, goal_position), 
                algo_index
            )
            frame_sequence_list.append(algorithm_frames)

        complete_frame_sequence = chain(*frame_sequence_list)

        # -------------------- VIDEO RENDERING --------------------
        export_figure: plt.Figure = plt.Figure(figsize=(10, 6))
        export_title_text: Text = export_figure.text(
            0.5, 0.95, "", ha='center', fontsize=14, weight='bold'
        )
        
        if should_include_maze_generation:
            export_maze_axes: Axes = export_figure.add_axes([0, 0, 1, 0.92])
            export_title_text.set_text("Maze Generation")
        else:
            export_maze_axes: Axes = export_figure.add_axes([0, 0.2, 1, 0.7])
        
        export_maze_axes.axis('off')
        export_table_axes: Axes = export_figure.add_axes([0.35, 0.17, 0.3, 0.25])
        export_table_axes.axis('off')
        
        export_image_display = export_maze_axes.imshow(
            next(complete_frame_sequence)["image"], cmap=MAZE_COLORMAP,
            vmin=0, vmax=len(CELL_COLORS)-1, aspect='auto'
        )
        
        self.current_export_algorithm = -1 
        self.export_results_table = None
        
        def update_export_frame(frame_data):
            """Update frame during video export."""
            export_image_display.set_data(frame_data["image"])

            algorithm_index = frame_data["algorithm_index"]
            if algorithm_index != self.current_export_algorithm:
                self.current_export_algorithm = algorithm_index
                
                if algorithm_index == -1:
                    export_title_text.set_text("Maze generation")
                else:
                    export_title_text.set_text(f"Running {selected_algorithms[self.current_export_algorithm].name} Algorithm")

                if algorithm_index == 0:
                    algorithm_names = [algo.name for algo in selected_algorithms]
                    self.export_results_table = export_table_axes.table(
                        cellText=[["-", "-", "-"] for _ in algorithm_names],
                        rowLabels=algorithm_names,
                        colLabels=["# Nodes", "Length", "Time[ms]"],
                        loc='bottom'
                    )
                    self.export_results_table.scale(1, 1.5)
                    export_maze_axes.set_position([0, 0.2, 1, 0.7])

            if algorithm_index >= 0:
                table_row = algorithm_index + 1
                if not np.isnan(frame_data["nodes_visited"]):
                    self.export_results_table[(table_row, 0)].get_text().set_text(int(frame_data["nodes_visited"]))
                if not np.isnan(frame_data["path_length"]):
                    self.export_results_table[(table_row, 1)].get_text().set_text(int(frame_data["path_length"]))
                if not np.isnan(frame_data["execution_time"]):
                    self.export_results_table[(table_row, 2)].get_text().set_text(int(frame_data["execution_time"]))

            return export_image_display, self.export_results_table

        animation_delay = self.animation_delay_slider.get()
        export_animation = FuncAnimation(
            export_figure, func=update_export_frame,
            frames=complete_frame_sequence, interval=animation_delay,
            blit=False, repeat=False, cache_frame_data=False
        )
        export_animation.pause()

        live_fps = np.mean(self.live_fps_list) if hasattr(self, 'live_fps_list') and self.live_fps_list else 30
        fps = min(1000 / animation_delay, live_fps)
        
        export_animation.save(output_filename, writer='ffmpeg', fps=fps)
        plt.close(export_figure)
        messagebox.showinfo("Export Complete", f"Animation saved to {output_filename}")

    # -------------------- APPLICATION LIFECYCLE --------------------
    def _on_application_close(self):
        """Clean up resources when application is closed."""
        if self.current_animation and self.current_animation.event_source is not None:
            self.current_animation.event_source.stop()
        plt.close('all')
        self.destroy()
        sys.exit(0)


# ---------------------------------------- APPLICATION ENTRY POINT ----------------------------------------
if __name__ == "__main__":
    pathfinding_application = GUI()
    pathfinding_application.mainloop()
