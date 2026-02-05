"""
Versión con visualización de red neuronal en tiempo real.
Integra el network visualization a la GUI existente.
"""
import os
import sys
import numpy as np
import torch
import threading
import time
import queue
from stable_baselines3 import PPO
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
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
import networkx as nx
from collections import deque
import matplotlib.colors as mcolors


class NeuralNetworkVisualizer:
    """Visualiza la red neuronal del modelo PPO"""
    
    def __init__(self, model):
        self.model = model
        self.policy_net = model.policy
        
        # Extraer arquitectura
        self.layers = []
        self.activations = []  # Guardará activaciones de cada capa
        
        # Obtener capas
        if hasattr(self.policy_net, 'mlp_extractor'):
            # Sacar información de MLP
            self.extract_mlp_layers()
        
    def extract_mlp_layers(self):
        """Extraer información de las capas MLP"""
        mlp = self.policy_net.mlp_extractor
        
        # Policy network
        if hasattr(mlp, 'policy_net'):
            for layer in mlp.policy_net:
                if isinstance(layer, torch.nn.Linear):
                    self.layers.append({
                        'type': 'linear',
                        'in_features': layer.in_features,
                        'out_features': layer.out_features,
                        'layer': layer
                    })
    
    def get_layer_activations(self, obs_input):
        """Obtener activaciones de cada capa durante forward pass"""
        activations = []
        
        # Forward pass con hook para capturar activaciones
        hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                activations.append(output.detach().cpu().numpy())
            return hook
        
        try:
            # Registrar hooks
            if hasattr(self.policy_net, 'mlp_extractor'):
                mlp = self.policy_net.mlp_extractor
                if hasattr(mlp, 'policy_net'):
                    for layer in mlp.policy_net:
                        if isinstance(layer, torch.nn.Linear):
                            hooks.append(layer.register_forward_hook(get_activation('linear')))
            
            # Forward pass
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs_input).unsqueeze(0)
                self.policy_net(obs_tensor)
            
            # Remover hooks
            for hook in hooks:
                hook.remove()
            
            return activations
        except Exception as e:
            print(f"Error obteniendo activaciones: {e}")
            return []


class NeuralGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Evaluador con Visualización Neural - Drones PyBullet")
        self.root.geometry("1800x950")
        
        # Variables
        self.model_path = tk.StringVar(value=os.path.join('results','obs19_nowind_withrandtarget_save-01.08.2026_16.11.20', 'best_model.zip'))
        self.is_running = False
        self.pause = False
        
        # Colas
        self.data_queue = queue.Queue(maxsize=100)
        self.neural_queue = queue.Queue(maxsize=50)
        
        # Buffers
        self.buffer_size = 300
        self.state_buffer = deque(maxlen=self.buffer_size)
        self.action_buffer = deque(maxlen=self.buffer_size)
        self.reward_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.neural_buffer = deque(maxlen=20)  # Últimas activaciones
        
        # Parámetros
        self.eval_params = {
            'multiagent': tk.BooleanVar(value=False),
            'gui': tk.BooleanVar(value=False),
            'record_video': tk.BooleanVar(value=False),
            'episodes': tk.IntVar(value=3),
            'max_steps': tk.IntVar(value=0),
            'speed_factor': tk.DoubleVar(value=1.0),
            'random_targets': tk.BooleanVar(value=True),
            'output_folder': tk.StringVar(value='results')
        }
        
        self.neural_visualizer = None
        self._create_gui()
        self._setup_update_loop()
        
    def _create_gui(self):
        """Crear interfaz"""
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel izquierdo - Controles
        left_frame = ttk.Frame(main_frame, width=280)
        main_frame.add(left_frame, weight=0)
        self._create_control_panel(left_frame)
        
        # Panel central - Gráficas dinámicas
        center_frame = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        main_frame.add(center_frame, weight=1)
        
        # Gráficas superiores
        plots_frame = ttk.Frame(center_frame)
        center_frame.add(plots_frame, height=400)
        self._create_plots_panel(plots_frame)
        
        # Red neuronal inferior
        neural_frame = ttk.LabelFrame(center_frame, text="Red Neuronal (Activaciones en tiempo real)", padding=5)
        center_frame.add(neural_frame, height=400)
        self._create_neural_panel(neural_frame)
        
    def _create_control_panel(self, parent):
        """Panel de controles"""
        model_frame = ttk.LabelFrame(parent, text="Modelo", padding=10)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Ruta Modelo:").pack(anchor=tk.W)
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path, width=20)
        model_entry.pack(fill=tk.X, pady=5)
        
        ttk.Button(model_frame, text="Seleccionar...", 
                  command=self._select_model).pack(fill=tk.X, pady=2)
        
        params_frame = ttk.LabelFrame(parent, text="Parámetros", padding=10)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Episodios:").pack(anchor=tk.W)
        ttk.Spinbox(params_frame, from_=1, to=50, textvariable=self.eval_params['episodes'],
                   width=10).pack(anchor=tk.W, pady=2)
        
        ttk.Label(params_frame, text="Pasos máximos:").pack(anchor=tk.W)
        ttk.Spinbox(params_frame, from_=0, to=10000, textvariable=self.eval_params['max_steps'],
                   width=10).pack(anchor=tk.W, pady=2)
        
        ttk.Label(params_frame, text="Factor velocidad:").pack(anchor=tk.W)
        self.speed_label = ttk.Label(params_frame, text="1.0x")
        self.speed_label.pack(anchor=tk.W)
        speed_scale = ttk.Scale(params_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL,
                               variable=self.eval_params['speed_factor'],
                               command=lambda v: self.speed_label.config(text=f"{float(v):.2f}x"))
        speed_scale.pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(params_frame, text="Objetivos aleatorios",
                       variable=self.eval_params['random_targets']).pack(anchor=tk.W, pady=5)
        
        options_frame = ttk.LabelFrame(parent, text="Opciones", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Checkbutton(options_frame, text="GUI PyBullet",
                       variable=self.eval_params['gui']).pack(anchor=tk.W, pady=2)
        
        control_frame = ttk.LabelFrame(parent, text="Control", padding=10)
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
        
        status_frame = ttk.LabelFrame(parent, text="Estado", padding=10)
        status_frame.pack(fill=tk.X, padx=5, pady=5, expand=True)
        
        self.status_label = ttk.Label(status_frame, text="Listo", wraplength=250)
        self.status_label.pack(anchor=tk.W)
        
        self.episode_label = ttk.Label(status_frame, text="Episodio: 0/0")
        self.episode_label.pack(anchor=tk.W)
        
        self.step_label = ttk.Label(status_frame, text="Paso: 0")
        self.step_label.pack(anchor=tk.W)
        
        self.reward_label = ttk.Label(status_frame, text="Recompensa: 0.0")
        self.reward_label.pack(anchor=tk.W)
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0")
        self.fps_label.pack(anchor=tk.W)
        
    def _create_plots_panel(self, parent):
        """Gráficas de dinámica del drone"""
        self.fig = Figure(figsize=(12, 4.5), dpi=90)
        self.fig.tight_layout(pad=2.5)
        
        self.ax_pos = self.fig.add_subplot(2, 3, 1)
        self.ax_vel = self.fig.add_subplot(2, 3, 2)
        self.ax_euler = self.fig.add_subplot(2, 3, 3)
        self.ax_actions = self.fig.add_subplot(2, 3, 4)
        self.ax_reward = self.fig.add_subplot(2, 3, 5)
        self.ax_target = self.fig.add_subplot(2, 3, 6)
        
        titles = ["Posición", "Velocidad", "Ángulos (°)", "Acciones (RPM)", 
                  "Recompensa", "Estado"]
        for ax, title in zip([self.ax_pos, self.ax_vel, self.ax_euler, self.ax_actions, 
                             self.ax_reward, self.ax_target], titles):
            ax.set_title(title, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Tiempo (s)", fontsize=8)
            ax.tick_params(labelsize=7)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _create_neural_panel(self, parent):
        """Visualización de red neuronal"""
        self.neural_fig = Figure(figsize=(12, 4), dpi=90)
        self.neural_ax = self.neural_fig.add_subplot(111)
        self.neural_ax.set_title("Activaciones de Neuronas en Tiempo Real", fontsize=10)
        self.neural_ax.axis('off')
        
        self.neural_canvas = FigureCanvasTkAgg(self.neural_fig, master=parent)
        self.neural_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _select_model(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar modelo PPO",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
            initialdir=os.path.join(os.getcwd(), 'results')
        )
        if filename:
            self.model_path.set(filename)
            
    def _start_evaluation(self):
        if not self.is_running:
            if not os.path.exists(self.model_path.get()):
                messagebox.showerror("Error", f"Archivo no encontrado: {self.model_path.get()}")
                return
            
            self.is_running = True
            self.pause = False
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            
            self.state_buffer.clear()
            self.action_buffer.clear()
            self.reward_buffer.clear()
            self.time_buffer.clear()
            self.neural_buffer.clear()
            
            eval_thread = threading.Thread(target=self._run_evaluation, daemon=True)
            eval_thread.start()
            
    def _toggle_pause(self):
        self.pause = not self.pause
        self.pause_btn.config(text="▶ Reanudar" if self.pause else "⏸ Pausar")
        
    def _stop_evaluation(self):
        self.is_running = False
        
    def _run_evaluation(self):
        """Evaluación en thread separado"""
        try:
            self.root.after(0, lambda: self.status_label.config(text="Inicializando..."))
            
            DEFAULT_OBS = ObservationType('kin')
            DEFAULT_ACT = ActionType('rpm')
            
            test_env = HoverAviary(
                gui=self.eval_params['gui'].get(),
                obs=DEFAULT_OBS,
                act=DEFAULT_ACT,
                record=self.eval_params['record_video'].get(),
                random_targets=self.eval_params['random_targets'].get(),
                physics=Physics.PYB
            )
            
            model = PPO.load(self.model_path.get(), env=test_env, device="cpu", verbose=0)
            self.neural_visualizer = NeuralNetworkVisualizer(model)
            
            num_episodes = self.eval_params['episodes'].get()
            max_steps = self.eval_params['max_steps'].get() or None
            speed_factor = self.eval_params['speed_factor'].get()
            
            for ep in range(num_episodes):
                if not self.is_running:
                    break
                
                self.root.after(0, lambda e=ep: self.episode_label.config(text=f"Episodio: {e+1}/{num_episodes}"))
                
                obs, info = test_env.reset(seed=ep, options={})
                start_time = time.time()
                total_reward = 0
                
                for step in range(max_steps or (test_env.EPISODE_LEN_SEC + 20) * test_env.CTRL_FREQ):
                    if not self.is_running:
                        break
                    
                    while self.pause and self.is_running:
                        time.sleep(0.01)
                    
                    # Predicción
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    obs2 = obs.squeeze()
                    act2 = action.squeeze()
                    total_reward += reward
                    
                    current_time = step / test_env.CTRL_FREQ
                    
                    # Datos de estado
                    try:
                        self.data_queue.put_nowait({
                            'time': current_time,
                            'pos': obs2[0:3].copy(),
                            'vel': obs2[3:6].copy(),
                            'euler': obs2[6:9].copy(),
                            'action': act2.copy() if hasattr(act2, 'copy') else float(act2),
                            'reward': total_reward,
                            'step': step,
                            'obs': obs.copy()
                        })
                    except queue.Full:
                        pass
                    
                    if hasattr(test_env, 'render'):
                        test_env.render()
                    
                    sync(step, start_time, test_env.CTRL_TIMESTEP * speed_factor)
                    
                    if terminated or truncated:
                        break
            
            test_env.close()
            self.root.after(0, lambda: self.status_label.config(text="Evaluación completada"))
            self.is_running = False
            self.root.after(0, self._reset_buttons)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error: {str(e)}"))
            self.is_running = False
            self.root.after(0, self._reset_buttons)
    
    def _reset_buttons(self):
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.pause_btn.config(text="⏸ Pausar")
    
    def _setup_update_loop(self):
        """Loop de actualización de GUI"""
        def update_loop():
            last_update = time.time()
            frame_count = 0
            
            while True:
                try:
                    while True:
                        try:
                            data = self.data_queue.get_nowait()
                            self.state_buffer.append({
                                'pos': data['pos'],
                                'vel': data['vel'],
                                'euler': data['euler'],
                                'ang_vel': np.zeros(3)
                            })
                            self.action_buffer.append(data['action'])
                            self.reward_buffer.append(data['reward'])
                            self.time_buffer.append(data['time'])
                            
                            # Obtener activaciones neuronales
                            if self.neural_visualizer and 'obs' in data:
                                try:
                                    activations = self.neural_visualizer.get_layer_activations(data['obs'])
                                    if activations:
                                        self.neural_buffer.append({
                                            'activations': activations,
                                            'input': data['obs'].squeeze()
                                        })
                                except:
                                    pass
                            
                            self.root.after(0, lambda s=data['step']: 
                                          self.step_label.config(text=f"Paso: {s}"))
                            self.root.after(0, lambda r=data['reward']: 
                                          self.reward_label.config(text=f"Recompensa: {r:.4f}"))
                            frame_count += 1
                        except queue.Empty:
                            break
                    
                    now = time.time()
                    if now - last_update > 0.1 and len(self.state_buffer) > 0:
                        fps = frame_count / (now - last_update)
                        self.root.after(0, lambda f=fps: self.fps_label.config(text=f"FPS: {f:.1f}"))
                        self.root.after(0, self._update_plots)
                        self.root.after(0, self._update_neural_vis)
                        last_update = now
                        frame_count = 0
                    
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(0.1)
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
    
    def _update_plots(self):
        """Actualizar gráficas"""
        if not self.time_buffer:
            return
        
        times = np.array(list(self.time_buffer))
        
        for ax in [self.ax_pos, self.ax_vel, self.ax_euler, self.ax_actions, self.ax_reward, self.ax_target]:
            ax.clear()
        
        if self.state_buffer:
            pos_data = np.array([s['pos'] for s in self.state_buffer])
            self.ax_pos.plot(times, pos_data[:, 0], 'b-', linewidth=1.5, label='X')
            self.ax_pos.plot(times, pos_data[:, 1], 'r-', linewidth=1.5, label='Y')
            self.ax_pos.plot(times, pos_data[:, 2], 'g-', linewidth=1.5, label='Z')
            self.ax_pos.legend(fontsize=7)
            self.ax_pos.grid(True, alpha=0.3)
        
        if self.state_buffer:
            vel_data = np.array([s['vel'] for s in self.state_buffer])
            self.ax_vel.plot(times, vel_data[:, 0], 'b-', linewidth=1.5, label='Vx')
            self.ax_vel.plot(times, vel_data[:, 1], 'r-', linewidth=1.5, label='Vy')
            self.ax_vel.plot(times, vel_data[:, 2], 'g-', linewidth=1.5, label='Vz')
            self.ax_vel.legend(fontsize=7)
            self.ax_vel.grid(True, alpha=0.3)
        
        if self.state_buffer:
            euler_data = np.array([s['euler'] for s in self.state_buffer])
            self.ax_euler.plot(times, np.degrees(euler_data[:, 0]), 'b-', linewidth=1.5, label='Roll')
            self.ax_euler.plot(times, np.degrees(euler_data[:, 1]), 'r-', linewidth=1.5, label='Pitch')
            self.ax_euler.plot(times, np.degrees(euler_data[:, 2]), 'g-', linewidth=1.5, label='Yaw')
            self.ax_euler.legend(fontsize=7)
            self.ax_euler.grid(True, alpha=0.3)
        
        if self.action_buffer:
            actions = np.array(self.action_buffer)
            if len(actions.shape) > 1:
                for i in range(min(actions.shape[1], 4)):
                    self.ax_actions.plot(times, actions[:, i], linewidth=1.5, label=f'M{i+1}')
            else:
                self.ax_actions.plot(times, actions, linewidth=1.5)
            self.ax_actions.legend(fontsize=7)
            self.ax_actions.grid(True, alpha=0.3)
        
        if self.reward_buffer:
            rewards = np.array(list(self.reward_buffer))
            self.ax_reward.plot(times, rewards, 'g-', linewidth=2)
            self.ax_reward.fill_between(times, rewards, alpha=0.3, color='green')
            self.ax_reward.grid(True, alpha=0.3)
        
        self.ax_target.text(0.5, 0.5, "GUI en tiempo real\ncon red neuronal", 
                          ha='center', va='center', transform=self.ax_target.transAxes, fontsize=10)
        
        self.fig.tight_layout(pad=2)
        self.canvas.draw_idle()
    
    def _update_neural_vis(self):
        """Actualizar visualización de red neuronal"""
        if not self.neural_buffer:
            return
        
        self.neural_ax.clear()
        
        latest = self.neural_buffer[-1]
        activations = latest['activations']
        input_data = latest['input']
        
        # Crear visualización
        y_offset = 0
        layer_height = 2
        
        # Entrada
        input_size = len(input_data)
        input_y_positions = np.linspace(0, layer_height, input_size)
        
        for i, val in enumerate(input_data):
            color_intensity = np.clip(np.abs(val), 0, 1)
            color = plt.cm.RdYlGn(color_intensity)
            circle = Circle((0, input_y_positions[i]), 0.3, color=color, ec='black', linewidth=1)
            self.neural_ax.add_patch(circle)
            self.neural_ax.text(-0.5, input_y_positions[i], f'{val:.2f}', 
                              ha='right', va='center', fontsize=7)
        
        # Capas
        x_offset = 1.5
        prev_positions = input_y_positions
        
        for layer_idx, layer_acts in enumerate(activations):
            if isinstance(layer_acts, np.ndarray):
                layer_acts = layer_acts.squeeze()
            
            if len(layer_acts.shape) == 0:  # Escalar
                layer_acts = np.array([layer_acts])
            
            num_neurons = len(layer_acts)
            neuron_y_positions = np.linspace(0, layer_height, num_neurons)
            
            # Dibujar conexiones
            for i, act_val in enumerate(layer_acts):
                # Peso promedio de conexiones anteriores
                for j, prev_pos in enumerate(prev_positions):
                    alpha = min(np.abs(act_val), 1.0) * 0.5
                    line_width = max(0.5, np.abs(act_val) * 3)
                    color_intensity = np.clip(np.abs(act_val), 0, 1)
                    
                    self.neural_ax.plot([x_offset - 1.5, x_offset], 
                                       [prev_pos, neuron_y_positions[i]],
                                       color=plt.cm.RdYlGn(color_intensity),
                                       linewidth=line_width, alpha=alpha)
            
            # Dibujar neuronas
            for i, act_val in enumerate(layer_acts):
                color_intensity = np.clip(np.abs(act_val), 0, 1)
                color = plt.cm.RdYlGn(color_intensity)
                size = 0.25 + np.abs(act_val) * 0.25
                circle = Circle((x_offset, neuron_y_positions[i]), size, 
                              color=color, ec='black', linewidth=1.5)
                self.neural_ax.add_patch(circle)
                self.neural_ax.text(x_offset, neuron_y_positions[i], f'{act_val:.1f}', 
                                  ha='center', va='center', fontsize=6, weight='bold')
            
            prev_positions = neuron_y_positions
            x_offset += 1.5
        
        self.neural_ax.set_xlim(-1.5, x_offset)
        self.neural_ax.set_ylim(-0.5, layer_height + 0.5)
        self.neural_ax.set_aspect('equal')
        self.neural_ax.axis('off')
        
        self.neural_fig.tight_layout()
        self.neural_canvas.draw_idle()


def main():
    root = tk.Tk()
    gui = NeuralGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
