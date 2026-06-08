# positional_encoding_study.py
import os
import torch
import math
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

os.makedirs("homework11/figures", exist_ok=True)

# ========================== Task 1: Sinusoidal Position Encoding ==========================
def compute_positional_encoding(sequence_length: int, hidden_dim: int) -> torch.Tensor:
    """Generate sinusoidal positional encodings for transformer models"""
    encoding_matrix = torch.zeros(sequence_length, hidden_dim)
    positions = torch.arange(sequence_length).unsqueeze(1)
    
    # Using standard sinusoidal formulation from "Attention Is All You Need"
    frequency_scale = 10000.0
    angle_rates = 1.0 / (frequency_scale ** (torch.arange(0, hidden_dim, 2) / hidden_dim))
    
    encoding_matrix[:, 0::2] = torch.sin(positions * angle_rates)
    encoding_matrix[:, 1::2] = torch.cos(positions * angle_rates)
    
    return encoding_matrix

def visualize_sinusoidal_encoding(max_positions=50, embedding_dim=128):
    """Create heatmap visualization of sinusoidal positional encoding"""
    pe_matrix = compute_positional_encoding(max_positions, embedding_dim)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap visualization
    im1 = axes[0].imshow(pe_matrix.numpy(), cmap='plasma', aspect='auto')
    axes[0].set_title('Sinusoidal Positional Encoding', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature Dimension', fontsize=12)
    axes[0].set_ylabel('Token Position', fontsize=12)
    plt.colorbar(im1, ax=axes[0])
    
    # Plot selected dimensions across positions
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for idx, dim in enumerate([0, 1, 2, 3, 4, 8, 16, 32]):
        if dim < embedding_dim:
            axes[1].plot(range(max_positions), pe_matrix[:, dim].numpy(), 
                       color=colors[idx], label=f'dim={dim}', linewidth=2)
    
    axes[1].set_title('Encoding Values Across Positions', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Position Index', fontsize=12)
    axes[1].set_ylabel('Encoding Value', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('homework11/figures/sinusoidal_encoding.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Task 1: Sinusoidal encoding visualized -> sinusoidal_encoding.png")


# ========================== Task 2: 2D Rotation Demonstration ==========================
def rotate_vector_2d(x_component: float, y_component: float, angle_rad: float) -> tuple:
    """Apply 2D rotation transformation to a vector"""
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    x_rotated = x_component * cos_theta - y_component * sin_theta
    y_rotated = x_component * sin_theta + y_component * cos_theta
    return x_rotated, y_rotated

def visualize_2d_rotation():
    """Illustrate how 2D rotation works for RoPE intuition"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # Left subplot: Multiple rotations of a single vector
    ax1 = axes[0]
    original_x, original_y = 1.0, 0.5
    ax1.arrow(0, 0, original_x, original_y, head_width=0.05, head_length=0.08,
             fc='darkred', ec='darkred', label='Original', linewidth=2)
    
    rotation_angles = [np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, np.pi]
    angle_names = ['30°', '60°', '90°', '120°', '180°']
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(rotation_angles)))
    
    for angle, name, color in zip(rotation_angles, angle_names, colors):
        x_rot, y_rot = rotate_vector_2d(original_x, original_y, angle)
        ax1.arrow(0, 0, x_rot, y_rot, head_width=0.05, head_length=0.08,
                 fc=color, ec=color, alpha=0.7, label=f'{name}')
    
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_title('2D Rotation: Vector at Different Angles', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Component', fontsize=12)
    ax1.set_ylabel('Y Component', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.2)
    
    # Right subplot: Rotation on unit circle
    ax2 = axes[1]
    unit_circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.5, color='blue')
    ax2.add_patch(unit_circle)
    
    test_angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, np.pi]
    for ang in test_angles:
        x_end, y_end = math.cos(ang), math.sin(ang)
        ax2.plot([0, x_end], [0, y_end], 'b-', alpha=0.3)
        ax2.plot(x_end, y_end, 'ro', markersize=6)
        degree = int(np.degrees(ang))
        ax2.annotate(f'{degree}°', (x_end*1.08, y_end*1.08), fontsize=9)
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax2.set_title('Geometric Interpretation: Rotation on Unit Circle', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Axis', fontsize=12)
    ax2.set_ylabel('Y Axis', fontsize=12)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('homework11/figures/2d_rotation_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Task 2: 2D rotation visualized -> 2d_rotation_demo.png")


# ========================== Task 3: High-dimensional RoPE ==========================
def apply_rotary_position_embedding(sequence: torch.Tensor, positions: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """
    Apply Rotary Position Embedding (RoPE) to input tensor
    RoPE rotates pairs of dimensions using position-dependent angles
    """
    assert feature_dim % 2 == 0, "Feature dimension must be even for RoPE"
    
    dimension_indices = torch.arange(0, feature_dim, 2, dtype=torch.float32)
    base_frequency = 10000.0
    
    # Compute rotation angles: theta_i = position * (base_freq)^{-2i/d}
    angular_freq = 1.0 / (base_frequency ** (dimension_indices / feature_dim))
    rotation_angles = positions.unsqueeze(1) * angular_freq.unsqueeze(0)
    
    # Split even and odd dimensions
    even_dim_vals = sequence[..., 0::2]
    odd_dim_vals = sequence[..., 1::2]
    
    cos_vals = torch.cos(rotation_angles)
    sin_vals = torch.sin(rotation_angles)
    
    # Apply rotation transformation
    rotated_even = even_dim_vals * cos_vals - odd_dim_vals * sin_vals
    rotated_odd = even_dim_vals * sin_vals + odd_dim_vals * cos_vals
    
    # Interleave back to original dimension order
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)

def visualize_rope_highdim(embedding_dim=128, seq_length=60):
    """Generate visualization of RoPE outputs across positions and dimensions"""
    # Create dummy input sequence (all ones for clarity)
    dummy_input = torch.ones(1, seq_length, embedding_dim)
    position_ids = torch.arange(seq_length)
    
    rope_output = apply_rotary_position_embedding(dummy_input, position_ids, embedding_dim)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap of RoPE outputs
    im1 = axes[0].imshow(rope_output[0].detach().numpy(), cmap='RdBu', aspect='auto', vmin=-1.2, vmax=1.2)
    axes[0].set_title('RoPE Feature Map (High-Dimensional)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature Dimension Index', fontsize=12)
    axes[0].set_ylabel('Sequence Position', fontsize=12)
    plt.colorbar(im1, ax=axes[0])
    
    # Show pattern for specific dimensions
    axes[1].set_title('RoPE Values for Selected Dimension Pairs', fontsize=14, fontweight='bold')
    
    dim_pairs = [(0, 1), (2, 3), (8, 9), (16, 17)]
    for idx, (dim_even, dim_odd) in enumerate(dim_pairs):
        if dim_odd < embedding_dim:
            pattern_even = rope_output[0, :, dim_even].numpy()
            pattern_odd = rope_output[0, :, dim_odd].numpy()
            axes[1].plot(range(seq_length), pattern_even, '--', linewidth=2, 
                        label=f'dim {dim_even} (even)')
            axes[1].plot(range(seq_length), pattern_odd, '-', linewidth=2, 
                        label=f'dim {dim_odd} (odd)', alpha=0.7)
    
    axes[1].set_xlabel('Position Index', fontsize=12)
    axes[1].set_ylabel('Encoding Value', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('homework11/figures/rope_highdim_output.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Task 3: High-dimensional RoPE visualized -> rope_highdim_output.png")


# ========================== Task 4: E+pos vs RoPE Comparison ==========================
def compare_encoding_methods(embedding_dim=96, seq_length=45):
    """Visual comparison between additive (E+pos) and rotary (RoPE) encoding methods"""
    # Create base embedding (constant values to see encoding effect clearly)
    base_embed = torch.ones(1, seq_length, embedding_dim) * 2.0
    
    # Generate sinusoidal positional encoding
    pos_encoding = compute_positional_encoding(seq_length, embedding_dim).unsqueeze(0)
    
    # Method 1: Additive encoding (E + pos)
    additive_output = (base_embed + pos_encoding)[0].numpy()
    
    # Method 2: Rotary encoding (RoPE)
    position_tensor = torch.arange(seq_length)
    rope_output = apply_rotary_position_embedding(base_embed, position_tensor, embedding_dim)[0].numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Additive encoding heatmap
    im1 = axes[0, 0].imshow(additive_output, cmap='coolwarm', vmin=-1.5, vmax=2.5, aspect='auto')
    axes[0, 0].set_title('Method A: E + Pos (Additive)', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Dimension', fontsize=11)
    axes[0, 0].set_ylabel('Position', fontsize=11)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # RoPE heatmap
    im2 = axes[0, 1].imshow(rope_output, cmap='coolwarm', vmin=-1.5, vmax=1.5, aspect='auto')
    axes[0, 1].set_title('Method B: RoPE (Rotary)', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Dimension', fontsize=11)
    axes[0, 1].set_ylabel('Position', fontsize=11)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Pattern comparison: take row at middle position
    middle_pos = seq_length // 2
    axes[1, 0].plot(range(embedding_dim), additive_output[middle_pos, :], 'b-', linewidth=1.5, label='E+Pos')
    axes[1, 0].plot(range(embedding_dim), rope_output[middle_pos, :], 'r-', linewidth=1.5, label='RoPE')
    axes[1, 0].set_title(f'Encoding Pattern at Position {middle_pos}', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Dimension Index', fontsize=11)
    axes[1, 0].set_ylabel('Encoding Value', fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistical comparison
    axes[1, 1].hist(additive_output.flatten(), bins=50, alpha=0.5, label='E+Pos', color='blue')
    axes[1, 1].hist(rope_output.flatten(), bins=50, alpha=0.5, label='RoPE', color='red')
    axes[1, 1].set_title('Value Distribution Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Encoding Value', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Position Encoding Methods', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('homework11/figures/encoding_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Task 4: E+pos vs RoPE comparison -> encoding_comparison.png")


# ========================== Task 5: Relative Position Property Validation ==========================
def validate_relative_position_property(feature_dim=64):
    """
    Empirical verification that RoPE's dot product depends only on relative position
    Using identical query and key vectors at different absolute positions
    """
    # Create identical query and key vectors
    query_vec = torch.ones(1, 1, feature_dim)
    key_vec = torch.ones(1, 1, feature_dim)
    
    test_cases = []
    
    # Case 1: Same relative distance (2) but different absolute positions
    q_pos1 = torch.tensor([1])
    k_pos1 = torch.tensor([3])  # relative = 2
    q_rot1 = apply_rotary_position_embedding(query_vec, q_pos1, feature_dim)
    k_rot1 = apply_rotary_position_embedding(key_vec, k_pos1, feature_dim)
    score1 = (q_rot1 @ k_rot1.transpose(-1, -2)).item()
    test_cases.append(('Δ=2 (pos 1→3)', score1))
    
    q_pos2 = torch.tensor([5])
    k_pos2 = torch.tensor([7])  # relative = 2 (same)
    q_rot2 = apply_rotary_position_embedding(query_vec, q_pos2, feature_dim)
    k_rot2 = apply_rotary_position_embedding(key_vec, k_pos2, feature_dim)
    score2 = (q_rot2 @ k_rot2.transpose(-1, -2)).item()
    test_cases.append(('Δ=2 (pos 5→7)', score2))
    
    # Case 2: Different relative distances
    q_pos3 = torch.tensor([0])
    k_pos3 = torch.tensor([5])  # relative = 5
    q_rot3 = apply_rotary_position_embedding(query_vec, q_pos3, feature_dim)
    k_rot3 = apply_rotary_position_embedding(key_vec, k_pos3, feature_dim)
    score3 = (q_rot3 @ k_rot3.transpose(-1, -2)).item()
    test_cases.append(('Δ=5 (pos 0→5)', score3))
    
    q_pos4 = torch.tensor([2])
    k_pos4 = torch.tensor([8])  # relative = 6
    q_rot4 = apply_rotary_position_embedding(query_vec, q_pos4, feature_dim)
    k_rot4 = apply_rotary_position_embedding(key_vec, k_pos4, feature_dim)
    score4 = (q_rot4 @ k_rot4.transpose(-1, -2)).item()
    test_cases.append(('Δ=6 (pos 2→8)', score4))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of scores
    labels = [case[0] for case in test_cases]
    scores = [case[1] for case in test_cases]
    
    colors_bar = ['#2ecc71' if i < 2 else '#e74c3c' for i in range(len(scores))]
    bars = axes[0].bar(labels, scores, color=colors_bar, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Dot Product Value', fontsize=12)
    axes[0].set_title('RoPE Relative Position Property', fontsize=14, fontweight='bold')
    axes[0].axhline(y=scores[0], color='green', linestyle='--', alpha=0.7, label='Δ=2 reference')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Table showing relative positions
    table_data = [
        ['Query Position', 'Key Position', 'Relative Distance', 'Dot Product'],
        ['1', '3', '2', f'{scores[0]:.6f}'],
        ['5', '7', '2', f'{scores[1]:.6f}'],
        ['0', '5', '5', f'{scores[2]:.6f}'],
        ['2', '8', '6', f'{scores[3]:.6f}'],
    ]
    
    axes[1].axis('tight')
    axes[1].axis('off')
    table = axes[1].table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Color the header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1].set_title('Experimental Verification', fontsize=14, fontweight='bold', y=0.98)
    
    plt.suptitle('RoPE: Dot Product Depends Only on Relative Position', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('homework11/figures/relative_position_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Task 5: Relative position validation -> relative_position_validation.png")
    print(f"\n  Verification results:")
    print(f"    - Δ=2 scores: {scores[0]:.6f} and {scores[1]:.6f} (should be equal)")
    print(f"    - Δ=5 score: {scores[2]:.6f} (different from Δ=2)")
    print(f"    - Δ=6 score: {scores[3]:.6f} (different from Δ=2 and Δ=5)")
    
    return scores


# ========================== Main Execution ==========================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("POSITIONAL ENCODING STUDY: Sinusoidal vs RoPE")
    print("="*60 + "\n")
    
    # Run all tasks
    visualize_sinusoidal_encoding()
    visualize_2d_rotation()
    visualize_rope_highdim()
    compare_encoding_methods()
    validate_relative_position_property()
    
    print("\n" + "="*60)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print(f"Output directory: homework11/figures/")
    print("="*60)