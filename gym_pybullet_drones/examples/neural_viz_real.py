"""
Visualizador de Red Neuronal con modelo real PPO.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import pygame
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics


class ModelActivationCapture:
    """Captura activaciones de un modelo PPO"""
    
    def __init__(self, model):
        self.model = model
        self.activations = []
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Registrar hooks para capturar activaciones"""
        def get_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations.append({
                        'name': name,
                        'data': output.detach().cpu().numpy().squeeze()
                    })
            return hook
        
        policy = self.model.policy
        layer_idx = 0
        
        if hasattr(policy, 'mlp_extractor'):
            mlp = policy.mlp_extractor
            if hasattr(mlp, 'policy_net'):
                for layer in mlp.policy_net:
                    if isinstance(layer, torch.nn.Linear):
                        name = f"policy_linear_{layer_idx}"
                        hook = layer.register_forward_hook(get_hook(name))
                        self.hooks.append(hook)
                        layer_idx += 1
        
        if hasattr(policy, 'action_net') and isinstance(policy.action_net, torch.nn.Linear):
            name = f"action_net_{layer_idx}"
            hook = policy.action_net.register_forward_hook(get_hook(name))
            self.hooks.append(hook)
    
    def forward(self, obs):
        """Forward pass y captura de activaciones"""
        self.activations = []
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.model.predict(obs_tensor, deterministic=True)
        return action, self.activations
    
    def cleanup(self):
        """Remover hooks"""
        for hook in self.hooks:
            hook.remove()


class NeuralNetVisualizer:
    """Visualizador de red neuronal en tiempo real"""
    
    def __init__(self, model_path):
        pygame.init()
        self.screen = pygame.display.set_mode((1600, 800), pygame.RESIZABLE)
        pygame.display.set_caption("Red Neuronal PPO - Tiempo Real")
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.model = PPO.load(model_path, device=("cuda" if torch.cuda.is_available() else "cpu"), verbose=0)
        try:
            if torch.cuda.is_available():
                from torch.cuda.amp import autocast
                orig_forward = self.model.policy.forward
                def _amp_forward(*args, **kwargs):
                    with autocast(enabled=True):
                        return orig_forward(*args, **kwargs)
                self.model.policy.forward = _amp_forward
        except Exception:
            pass
        self.model.policy.eval()
        self.activation_capture = ModelActivationCapture(self.model)
        
        self._extract_architecture()
        
        self.font_small = pygame.font.Font(None, 16)
        self.current_activations = []
        self.current_input = None
    
    def _extract_architecture(self):
        """Extraer arquitectura del modelo"""
        obs_space = self.model.observation_space
        shape = obs_space.shape
        input_size = shape[-1] if isinstance(shape, tuple) else shape
        
        self.layer_dims = [input_size]
        
        policy = self.model.policy
        if hasattr(policy, 'mlp_extractor') and hasattr(policy.mlp_extractor, 'policy_net'):
            for layer in policy.mlp_extractor.policy_net:
                if isinstance(layer, torch.nn.Linear):
                    self.layer_dims.append(layer.out_features)
        
        if hasattr(policy, 'action_net') and isinstance(policy.action_net, torch.nn.Linear):
            if self.layer_dims[-1] != policy.action_net.out_features:
                self.layer_dims.append(policy.action_net.out_features)
        
        print(f"✓ Arquitectura: {self.layer_dims}")
    
    def _normalize_value(self, value):
        """Normalizar a [-1, 1]"""
        if isinstance(value, np.ndarray):
            value = float(value.flatten()[0])
        return float(np.clip(value, -1, 1))
    
    def _get_color(self, value):
        """Color según activación"""
        value = self._normalize_value(value)
        
        if value > 0.3:
            intensity = value
            return (0, int(255 * intensity), int(100 * intensity))
        elif value > 0:
            intensity = value
            return (int(255 * intensity), int(255 * intensity), 100)
        elif value > -0.3:
            intensity = abs(value)
            return (int(150 + 50 * intensity), int(150 + 50 * intensity), int(150 + 50 * intensity))
        else:
            intensity = abs(value)
            return (int(255 * intensity), int(50 * intensity), int(50 * intensity))
    
    def _draw_neuron(self, x, y, value, radius=12):
        """Dibujar neurona"""
        norm_val = abs(self._normalize_value(value))
        color = self._get_color(value)
        size = radius + radius * norm_val * 0.4
        
        pygame.draw.circle(self.screen, color, (int(x), int(y)), int(size))
        pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), int(size), 2)
        
        try:
            text = self.font_small.render(f"{value:.1f}", True, (0, 0, 0))
            self.screen.blit(text, text.get_rect(center=(int(x), int(y))))
        except:
            pass
    
    def _draw_connection(self, x1, y1, x2, y2, weight):
        """Dibujar conexión"""
        thickness = max(1, int(abs(weight) * 6))
        color = (100, 200, 100) if weight > 0 else (200, 100, 100)
        pygame.draw.line(self.screen, color, (int(x1), int(y1)), (int(x2), int(y2)), thickness)
    
    def draw(self):
        """Dibujar red neuronal"""
        self.screen.fill((15, 15, 25))
        
        h, w = self.screen.get_height(), self.screen.get_width()
        margin_x, margin_y = 60, 80
        
        layer_x = np.linspace(margin_x, w - margin_x, len(self.layer_dims))
        
        # Conexiones
        for layer_idx in range(len(self.layer_dims) - 1):
            curr_y = np.linspace(margin_y, h - margin_y, self.layer_dims[layer_idx])
            next_y = np.linspace(margin_y, h - margin_y, self.layer_dims[layer_idx + 1])
            
            for i in range(self.layer_dims[layer_idx]):
                for j in range(self.layer_dims[layer_idx + 1]):
                    self._draw_connection(layer_x[layer_idx], curr_y[i],
                                        layer_x[layer_idx + 1], next_y[j], 0.5)
        
        # Neuronas
        for layer_idx in range(len(self.layer_dims)):
            y_pos = np.linspace(margin_y, h - margin_y, self.layer_dims[layer_idx])
            
            for i, y in enumerate(y_pos):
                if layer_idx == 0:
                    value = self.current_input[i] if self.current_input is not None and i < len(self.current_input) else 0
                else:
                    act_idx = layer_idx - 1
                    if act_idx < len(self.current_activations):
                        act = self.current_activations[act_idx]
                        data = act['data'] if isinstance(act, dict) else act
                        value = float(data[i]) if hasattr(data, '__len__') and i < len(data) else 0
                    else:
                        value = 0
                
                self._draw_neuron(layer_x[layer_idx], y, value)
        
        # Etiquetas
        for i, x in enumerate(layer_x):
            if i == 0:
                label = f"INPUT\n({self.layer_dims[i]})"
            elif i == len(self.layer_dims) - 1:
                label = f"OUTPUT\n({self.layer_dims[i]})"
            else:
                label = f"HIDDEN\n({self.layer_dims[i]})"
            
            text = self.font_small.render(label, True, (220, 220, 220))
            self.screen.blit(text, text.get_rect(center=(int(x), 30)))
        
        info = self.font_small.render(f"FPS: {self.clock.get_fps():.1f}", True, (220, 220, 220))
        self.screen.blit(info, (10, h - 40))
    
    def update(self, obs):
        """Actualizar con observación"""
        self.current_input = obs
        action, activations = self.activation_capture.forward(obs)
        self.current_activations = activations
    
    def run(self):
        """Ejecutar con ambiente real"""
        env = HoverAviary(gui=False, obs=ObservationType('kin'), act=ActionType('rpm'), physics=Physics.PYB)
        obs, _ = env.reset()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.running = False
            
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            
            self.update(obs.squeeze())
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        env.close()
        self.cleanup()
    
    def cleanup(self):
        """Limpiar recursos"""
        self.activation_capture.cleanup()
        pygame.quit()


def main():
    model_path = os.path.join('results', 'obs12_8_lidar_nowind_randtarget_save-02.27.2026_18.19.22', 'best_model.zip')
    
    if not os.path.exists(model_path):
        print(f"Error: Modelo no encontrado en {model_path}")
        return
    
    print(f"Cargando modelo: {model_path}")
    viz = NeuralNetVisualizer(model_path)
    viz.run()


if __name__ == '__main__':
    main()
