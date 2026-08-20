import os
import sys
import numpy as np
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
import threading
import time
from stable_baselines3 import PPO
import glob
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics
from gym_pybullet_drones.utils.utils import sync

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from collections import deque


class EvalModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Evaluador de Modelo - Drones Gym PyBullet")
        self.root.geometry("1600x900")
        
        # Variables de control
        self.model_path = tk.StringVar(value=os.path.join('results','obs19_nowind_withrandtarget_save-01.08.2026_16.11.20', 'best_model.zip'))
        self.is_running = False
        self.pause = False
        self.episode_num = 0
        self.step_num = 0
        
        # Bufferes para gráficas en tiempo real (últimos 500 puntos)
        self.buffer_size = 500
        self.state_buffer = deque(maxlen=self.buffer_size)
        self.action_buffer = deque(maxlen=self.buffer_size)
        self.reward_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        
        # Control de actualización de gráficas (actualizar cada N pasos)
        self.plot_update_interval = 5
        self.plot_step_counter = 0
        
        # Parámetros de evaluación
        self.eval_params = {
            'multiagent': tk.BooleanVar(value=False),
            'gui': tk.BooleanVar(value=True),
            'record_video': tk.BooleanVar(value=False),
            'episodes': tk.IntVar(value=3),
            'max_steps': tk.IntVar(value=0),  # 0 = sin límite
            'speed_factor': tk.DoubleVar(value=1.0),
            'random_targets': tk.BooleanVar(value=True),
            'output_folder': tk.StringVar(value='results')
        }
        
        self._create_gui()
        # Poblar la lista de modelos al iniciar
        try:
            self._refresh_model_list()
        except Exception:
            pass
        
    def _create_gui(self):
        """Crear la interfaz gráfica"""
        # Frame principal
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel izquierdo - Controles
        left_frame = ttk.Frame(main_frame, width=300)
        main_frame.add(left_frame, weight=0)
        self._create_control_panel(left_frame)
        
        # Panel derecho - Gráficas
        right_frame = ttk.Frame(main_frame)
        main_frame.add(right_frame, weight=1)
        self._create_plots_panel(right_frame)
        
    def _create_control_panel(self, parent):
        """Crear panel de controles"""
        # Scrollable frame
        canvas = tk.Canvas(parent, bg='white')
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # === SECCIÓN MODELO ===
        model_frame = ttk.LabelFrame(scrollable_frame, text="Modelo", padding=10)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Ruta Modelo:").pack(anchor=tk.W)
        # Combobox con la lista de modelos encontrados en la carpeta de resultados
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_path, width=60)
        self.model_combo.pack(fill=tk.X, pady=5)
        btns_frame = ttk.Frame(model_frame)
        btns_frame.pack(fill=tk.X)
        ttk.Button(btns_frame, text="Seleccionar...", command=self._select_model).pack(side=tk.LEFT, fill=tk.X, expand=True, pady=2, padx=(0,5))
        ttk.Button(btns_frame, text="Refrescar lista", command=self._refresh_model_list).pack(side=tk.LEFT, fill=tk.X, expand=True, pady=2)
        
        # === SECCIÓN PARÁMETROS ===
        params_frame = ttk.LabelFrame(scrollable_frame, text="Parámetros", padding=10)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Episodios
        ttk.Label(params_frame, text="Episodios:").pack(anchor=tk.W)
        ttk.Spinbox(params_frame, from_=1, to=50, textvariable=self.eval_params['episodes'],
                   width=10).pack(anchor=tk.W, pady=2)
        
        # Pasos máximo
        ttk.Label(params_frame, text="Pasos máximos (0=sin límite):").pack(anchor=tk.W)
        ttk.Spinbox(params_frame, from_=0, to=10000, textvariable=self.eval_params['max_steps'],
                   width=10).pack(anchor=tk.W, pady=2)
        
        # Velocidad
        ttk.Label(params_frame, text="Factor de velocidad:").pack(anchor=tk.W)
        speed_scale = ttk.Scale(params_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL,
                               variable=self.eval_params['speed_factor'])
        speed_scale.pack(fill=tk.X, pady=2)
        speed_label = ttk.Label(params_frame, text="1.0x")
        speed_label.pack(anchor=tk.W)
        
        def update_speed_label(val):
            speed_label.config(text=f"{float(val):.2f}x")
        speed_scale.config(command=update_speed_label)
        
        # Objetivos aleatorios
        ttk.Checkbutton(params_frame, text="Objetivos aleatorios",
                       variable=self.eval_params['random_targets']).pack(anchor=tk.W, pady=5)
        
        # === SECCIÓN OPCIONES ===
        options_frame = ttk.LabelFrame(scrollable_frame, text="Opciones", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(options_frame, text="GUI PyBullet",
                       variable=self.eval_params['gui']).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(options_frame, text="Multi-agente",
                       variable=self.eval_params['multiagent']).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(options_frame, text="Grabar video",
                       variable=self.eval_params['record_video']).pack(anchor=tk.W, pady=2)
        
        # === SECCIÓN CARPETA SALIDA ===
        output_frame = ttk.LabelFrame(scrollable_frame, text="Salida", padding=10)
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(output_frame, text="Carpeta de resultados:").pack(anchor=tk.W)
        output_entry = ttk.Entry(output_frame, textvariable=self.eval_params['output_folder'], width=25)
        output_entry.pack(fill=tk.X, pady=5)
        
        # === BOTONES DE CONTROL ===
        control_frame = ttk.LabelFrame(scrollable_frame, text="Control", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_btn = ttk.Button(control_frame, text="▶ Iniciar", 
                                   command=self._start_evaluation)
        self.start_btn.pack(fill=tk.X, pady=2)
        
        self.pause_btn = ttk.Button(control_frame, text="⏸ Pausar", 
                                   command=self._toggle_pause, state=tk.DISABLED)
        self.pause_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹ Detener", 
                                  command=self._stop_evaluation, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        # === ESTADO ===
        status_frame = ttk.LabelFrame(scrollable_frame, text="Estado", padding=10)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Listo")
        self.status_label.pack(anchor=tk.W)
        
        self.episode_label = ttk.Label(status_frame, text="Episodio: 0")
        self.episode_label.pack(anchor=tk.W)
        
        self.step_label = ttk.Label(status_frame, text="Paso: 0")
        self.step_label.pack(anchor=tk.W)
        
        self.reward_label = ttk.Label(status_frame, text="Recompensa total: 0.0")
        self.reward_label.pack(anchor=tk.W)
        
        # Pack canvas y scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def _create_plots_panel(self, parent):
        """Crear panel de gráficas"""
        # Frame para las gráficas
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.fig.tight_layout(pad=3)
        
        # Subplots
        self.ax_pos = self.fig.add_subplot(2, 3, 1)
        self.ax_vel = self.fig.add_subplot(2, 3, 2)
        self.ax_euler = self.fig.add_subplot(2, 3, 3)
        self.ax_actions = self.fig.add_subplot(2, 3, 4)
        self.ax_reward = self.fig.add_subplot(2, 3, 5)
        self.ax_thrust = self.fig.add_subplot(2, 3, 6)
        
        # Títulos
        self.ax_pos.set_title("Posición (x, y, z)")
        self.ax_vel.set_title("Velocidad (vx, vy, vz)")
        self.ax_euler.set_title("Ángulos de Euler (roll, pitch, yaw)")
        self.ax_actions.set_title("Acciones (RPM promedio)")
        self.ax_reward.set_title("Recompensa Acumulada")
        self.ax_thrust.set_title("Thrust (F1, F2, F3, F4)")
        
        # Etiquetas
        for ax in [self.ax_pos, self.ax_vel, self.ax_euler, self.ax_actions, self.ax_reward, self.ax_thrust]:
            ax.set_xlabel("Tiempo (s)")
            ax.grid(True, alpha=0.3)
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _select_model(self):
        """Seleccionar archivo de modelo"""
        filename = filedialog.askopenfilename(
            title="Seleccionar modelo PPO",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
            initialdir=os.path.join(os.getcwd(), 'results')
        )
        if filename:
            self.model_path.set(filename)
            # Si usamos el combobox, asegurarnos de que la selección esté en la lista
            try:
                if hasattr(self, 'model_combo') and filename not in self.model_combo['values']:
                    vals = list(self.model_combo['values']) if self.model_combo['values'] else []
                    vals.insert(0, filename)
                    self.model_combo['values'] = vals
            except Exception:
                pass

    def _refresh_model_list(self):
        """Buscar modelos .zip en la carpeta de resultados y actualizar el combobox"""
        try:
            out_dir = self.eval_params['output_folder'].get() if 'output_folder' in self.eval_params else 'results'
            # Buscar recursivamente .zip
            pattern = os.path.join(out_dir, '**', '*.zip')
            files = glob.glob(pattern, recursive=True)
            # Añadir también búsqueda en 'results' si no encontrada
            if not files and out_dir != 'results':
                files = glob.glob(os.path.join('results', '**', '*.zip'), recursive=True)
            files = sorted(files, key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
            if hasattr(self, 'model_combo'):
                self.model_combo['values'] = files
                # Si hay elementos, seleccionar el primero si no hay valor actual
                if files and (not self.model_path.get() or self.model_path.get() not in files):
                    self.model_path.set(files[0])
        except Exception:
            pass
            
    def _start_evaluation(self):
        """Iniciar evaluación"""
        if not self.is_running:
            if not os.path.exists(self.model_path.get()):
                messagebox.showerror("Error", f"Archivo no encontrado: {self.model_path.get()}")
                return
            
            self.is_running = True
            self.pause = False
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Limpiar buffers
            self.state_buffer.clear()
            self.action_buffer.clear()
            self.reward_buffer.clear()
            self.time_buffer.clear()
            
            # Iniciar evaluación en thread separado
            eval_thread = threading.Thread(target=self._run_evaluation, daemon=True)
            eval_thread.start()
            
    def _toggle_pause(self):
        """Pausar/Reanudar"""
        self.pause = not self.pause
        self.pause_btn.config(text="▶ Reanudar" if self.pause else "⏸ Pausar")
        
    def _stop_evaluation(self):
        """Detener evaluación"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(text="⏸ Pausar")
        self.status_label.config(text="Detenido")
        
    def _run_evaluation(self):
        """Ejecutar evaluación (en thread separado)"""
        try:
            self.status_label.config(text="Inicializando...")
            
            DEFAULT_OBS = ObservationType('kin')
            DEFAULT_ACT = ActionType('rpm')
            DEFAULT_AGENTS = 1
            
            # Crear ambiente
            if not self.eval_params['multiagent'].get():
                test_env = HoverAviary(
                    gui=self.eval_params['gui'].get(),
                    obs=DEFAULT_OBS,
                    act=DEFAULT_ACT,
                    record=self.eval_params['record_video'].get(),
                    random_targets=self.eval_params['random_targets'].get(),
                    physics=Physics.PYB
                )
            else:
                test_env = MultiHoverAviary(
                    gui=self.eval_params['gui'].get(),
                    num_drones=DEFAULT_AGENTS,
                    obs=DEFAULT_OBS,
                    act=DEFAULT_ACT,
                    record=self.eval_params['record_video'].get()
                )
            
            # Cargar modelo
            model = PPO.load(self.model_path.get(), env=test_env, device=DEVICE, verbose=0)
            try:
                if DEVICE == 'cuda':
                    from torch.cuda.amp import autocast
                    orig_forward = model.policy.forward
                    def _amp_forward(*args, **kwargs):
                        with autocast(enabled=True):
                            return orig_forward(*args, **kwargs)
                    model.policy.forward = _amp_forward
            except Exception:
                pass
            
            num_episodes = self.eval_params['episodes'].get()
            max_steps = self.eval_params['max_steps'].get()
            if max_steps == 0:
                max_steps = None
            speed_factor = self.eval_params['speed_factor'].get()
            
            # Ejecutar episodios
            for ep in range(num_episodes):
                if not self.is_running:
                    break
                    
                self.episode_num = ep + 1
                self.step_num = 0
                self.status_label.config(text=f"Ejecutando episodio {self.episode_num}/{num_episodes}")
                
                obs, info = test_env.reset(seed=ep, options={})
                start_time = time.time()
                total_reward = 0
                
                for step in range(max_steps or (test_env.EPISODE_LEN_SEC + 20) * test_env.CTRL_FREQ):
                    if not self.is_running:
                        break
                    
                    # Pausar si es necesario
                    while self.pause and self.is_running:
                        time.sleep(0.1)
                    
                    self.step_num = step
                    
                    # Predicción del modelo
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    
                    obs2 = obs.squeeze()
                    act2 = action.squeeze()
                    total_reward += reward
                    
                    # Guardar datos en buffers
                    current_time = (step / test_env.CTRL_FREQ)
                    self.time_buffer.append(current_time)
                    
                    # Estado: posición [0:3], velocidad [3:6], ángulos [6:9], velocidad angular [9:12]
                    state_data = {
                        'pos': obs2[0:3].copy(),
                        'vel': obs2[3:6].copy(),
                        'euler': obs2[6:9].copy(),
                        'ang_vel': obs2[9:12].copy()
                    }
                    self.state_buffer.append(state_data)
                    
                    # Acciones
                    self.action_buffer.append(act2.copy() if hasattr(act2, 'copy') else float(act2))
                    
                    # Recompensa
                    self.reward_buffer.append(total_reward)
                    
                    # Actualizar etiquetas
                    self.step_label.config(text=f"Paso: {self.step_num}")
                    self.reward_label.config(text=f"Recompensa total: {total_reward:.4f}")
                    
                    # Renderizar si está habilitado
                    if hasattr(test_env, 'render'):
                        test_env.render()
                    
                    # Sincronizar tiempo
                    sync(step, start_time, test_env.CTRL_TIMESTEP * speed_factor)
                    
                    # Actualizar gráficas cada N pasos (para mejor rendimiento)
                    self.plot_step_counter += 1
                    if self.plot_step_counter >= self.plot_update_interval:
                        self.plot_step_counter = 0
                        self.root.after(0, self._update_plots)
                    
                    if terminated or truncated:
                        break
                
                self.root.after(0, lambda: self.reward_label.config(
                    text=f"Recompensa total: {total_reward:.4f} (Episodio {ep+1} completado)"))
            
            test_env.close()
            self.status_label.config(text="Evaluación completada")
            self.is_running = False
            self.root.after(0, self._reset_buttons)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante la evaluación:\n{str(e)}")
            self.status_label.config(text="Error")
            self.is_running = False
            self.root.after(0, self._reset_buttons)
    
    def _reset_buttons(self):
        """Resetear botones después de evaluación"""
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(text="⏸ Pausar")
    
    def _update_plots(self):
        """Actualizar gráficas en tiempo real (optimizado)"""
        if not self.time_buffer or len(self.state_buffer) < 2:
            return
        
        times = np.array(list(self.time_buffer))
        
        # Limpiar ejes
        for ax in [self.ax_pos, self.ax_vel, self.ax_euler, self.ax_actions, self.ax_reward, self.ax_thrust]:
            ax.clear()
        
        # Posición
        if self.state_buffer:
            pos_data = np.array([s['pos'] for s in self.state_buffer])
            self.ax_pos.plot(times, pos_data[:, 0], 'b-', linewidth=1.5, label='X')
            self.ax_pos.plot(times, pos_data[:, 1], 'r-', linewidth=1.5, label='Y')
            self.ax_pos.plot(times, pos_data[:, 2], 'g-', linewidth=1.5, label='Z')
            self.ax_pos.set_title("Posición (x, y, z)", fontsize=9)
            self.ax_pos.legend(fontsize=7)
            self.ax_pos.grid(True, alpha=0.3)
            self.ax_pos.set_xlabel("Tiempo (s)", fontsize=7)
            self.ax_pos.tick_params(labelsize=7)
        
        # Velocidad
        if self.state_buffer:
            vel_data = np.array([s['vel'] for s in self.state_buffer])
            self.ax_vel.plot(times, vel_data[:, 0], 'b-', linewidth=1.5, label='Vx')
            self.ax_vel.plot(times, vel_data[:, 1], 'r-', linewidth=1.5, label='Vy')
            self.ax_vel.plot(times, vel_data[:, 2], 'g-', linewidth=1.5, label='Vz')
            self.ax_vel.set_title("Velocidad (vx, vy, vz)", fontsize=9)
            self.ax_vel.legend(fontsize=7)
            self.ax_vel.grid(True, alpha=0.3)
            self.ax_vel.set_xlabel("Tiempo (s)", fontsize=7)
            self.ax_vel.tick_params(labelsize=7)
        
        # Ángulos de Euler
        if self.state_buffer:
            euler_data = np.array([s['euler'] for s in self.state_buffer])
            self.ax_euler.plot(times, np.degrees(euler_data[:, 0]), 'b-', linewidth=1.5, label='Roll')
            self.ax_euler.plot(times, np.degrees(euler_data[:, 1]), 'r-', linewidth=1.5, label='Pitch')
            self.ax_euler.plot(times, np.degrees(euler_data[:, 2]), 'g-', linewidth=1.5, label='Yaw')
            self.ax_euler.set_title("Ángulos (°)", fontsize=9)
            self.ax_euler.legend(fontsize=7)
            self.ax_euler.grid(True, alpha=0.3)
            self.ax_euler.set_xlabel("Tiempo (s)", fontsize=7)
            self.ax_euler.tick_params(labelsize=7)
        
        # Acciones
        if self.action_buffer:
            actions = np.array(self.action_buffer)
            if len(actions.shape) > 1:
                for i in range(min(actions.shape[1], 4)):
                    self.ax_actions.plot(times, actions[:, i], linewidth=1.5, label=f'M{i+1}')
            else:
                self.ax_actions.plot(times, actions, linewidth=1.5, label='RPM')
            self.ax_actions.set_title("Acciones (RPM)", fontsize=9)
            self.ax_actions.legend(fontsize=7)
            self.ax_actions.grid(True, alpha=0.3)
            self.ax_actions.set_xlabel("Tiempo (s)", fontsize=7)
            self.ax_actions.tick_params(labelsize=7)
        
        # Recompensa acumulada
        if self.reward_buffer:
            reward_data = np.array(list(self.reward_buffer))
            self.ax_reward.plot(times, reward_data, 'g-', linewidth=2)
            self.ax_reward.fill_between(times, reward_data, alpha=0.3, color='green')
            self.ax_reward.set_title("Recompensa Acumulada", fontsize=9)
            self.ax_reward.grid(True, alpha=0.3)
            self.ax_reward.set_xlabel("Tiempo (s)", fontsize=7)
            self.ax_reward.tick_params(labelsize=7)
        
        # Thrust
        self.ax_thrust.text(0.5, 0.5, "Thrust data\nnot available\nin kinematic obs", 
                          ha='center', va='center', transform=self.ax_thrust.transAxes, fontsize=9)
        self.ax_thrust.set_title("Thrust", fontsize=9)
        
        self.fig.tight_layout(pad=2)
        self.canvas.draw_idle()  # Usar draw_idle() es más eficiente que draw()


def main():
    root = tk.Tk()
    gui = EvalModelGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
