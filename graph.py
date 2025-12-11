import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_beautiful_timeline():
    # 设置风格：极简、白底
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    
    # 定义高级灰配色
    c_user = '#BDC3C7'  # 浅灰 (User)
    c_fast = '#1ABC9C'  # 青绿 (Fast)
    c_slow = '#E8E8E8'  # 极浅灰 (Background process)
    c_gaze = '#F39C12'  # 橙色 (Behavior 1)
    c_fill = '#E67E22'  # 深橙 (Behavior 2)
    c_resp = '#3498DB'  # 蓝色 (Response)
    
    # -------------------------------------------------------
    # 1. 绘制背景层：Slow Model (画在最底下，表示它在后台)
    # -------------------------------------------------------
    # 用一个长条表示 Reasoning Window
    ax.add_patch(mpatches.FancyBboxPatch((0.6, 2.8), 4.0, 0.4, 
                                         boxstyle="round,pad=0.1", 
                                         fc=c_slow, ec='#95A5A6', lw=1, ls='--'))
    ax.text(2.6, 3.0, "Internal Reasoning Process (Latency: 4.5s)", 
            ha='center', va='center', color='#7F8C8D', fontsize=9, style='italic')

    # -------------------------------------------------------
    # 2. 绘制 User Input
    # -------------------------------------------------------
    ax.add_patch(mpatches.FancyBboxPatch((0.0, 4.0), 0.5, 0.6, 
                                         boxstyle="round,pad=0.1", fc=c_user, ec='none'))
    ax.text(0.25, 4.3, 'User Input', ha='center', va='center', color='white', fontweight='bold')
    ax.text(0.25, 3.8, '"Black hole?"', ha='center', va='center', color='#2C3E50', fontsize=9)

    # 箭头：User -> Fast
    ax.annotate("", xy=(0.55, 4.3), xytext=(0.6, 4.3), arrowprops=dict(arrowstyle="-", color="#BDC3C7"))

    # -------------------------------------------------------
    # 3. 绘制 Fast Model (Trigger)
    # -------------------------------------------------------
    # 画一个小竖条或小圆点表示触发点
    ax.plot([0.6, 0.6], [2.8, 4.8], color=c_fast, lw=2, alpha=0.6) # 竖线
    ax.text(0.6, 5.0, "Fast Model\n(Confidence < Threshold)", ha='center', va='bottom', color=c_fast, fontsize=8, fontweight='bold')

    # -------------------------------------------------------
    # 4. 绘制 Robot External Behaviors (Masking)
    # -------------------------------------------------------
    # Behavior 1: Gaze Aversion
    # y=1.5
    ax.add_patch(mpatches.FancyBboxPatch((0.7, 1.5), 1.5, 0.6, 
                                         boxstyle="round,pad=0.1", fc=c_gaze, ec='none', alpha=0.9))
    ax.text(1.45, 1.8, 'Gaze Aversion', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    ax.text(1.45, 1.3, '(Non-verbal)', ha='center', va='center', color=c_gaze, fontsize=8)

    # Behavior 2: Filler
    # 稍微错开一点，显得是序列
    ax.add_patch(mpatches.FancyBboxPatch((2.3, 1.5), 2.0, 0.6, 
                                         boxstyle="round,pad=0.1", fc=c_fill, ec='none', alpha=0.9))
    ax.text(3.3, 1.8, 'Verbal Filler', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    ax.text(3.3, 1.3, '"Hmm, let me think..."', ha='center', va='center', color=c_fill, fontsize=8)

    # -------------------------------------------------------
    # 5. 绘制 Final Response
    # -------------------------------------------------------
    ax.add_patch(mpatches.FancyBboxPatch((4.7, 1.5), 1.5, 0.6, 
                                         boxstyle="round,pad=0.1", fc=c_resp, ec='none'))
    ax.text(5.45, 1.8, 'Final Response', ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    
    # -------------------------------------------------------
    # 装饰与轴设置
    # -------------------------------------------------------
    # 画一条时间轴线在最底下
    ax.plot([0, 6.5], [0.8, 0.8], color='#333333', lw=1.5)
    # 时间刻度
    for t in [0, 1, 2, 3, 4, 5, 6]:
        ax.plot([t, t], [0.8, 0.7], color='#333333', lw=1.5)
        ax.text(t, 0.5, f"{t}s", ha='center', va='center', fontsize=9)
    
    ax.set_xlim(-0.2, 6.5)
    ax.set_ylim(0, 5.5)
    
    # 隐藏边框和轴
    ax.axis('off')
    
    # 添加左侧的标签 (Tracks)
    ax.text(-0.1, 4.3, "Input", ha='right', va='center', fontweight='bold', color='#7F8C8D')
    ax.text(-0.1, 3.0, "Internal", ha='right', va='center', fontweight='bold', color='#7F8C8D')
    ax.text(-0.1, 1.8, "Behavior", ha='right', va='center', fontweight='bold', color='#7F8C8D')

    plt.tight_layout()
    plt.show()

draw_beautiful_timeline()