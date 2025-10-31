import json
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import textwrap
from matplotlib.patches import Rectangle, FancyBboxPatch
from collections import defaultdict
import matplotlib.colors as mcolors
import os

def visualize_hypotheses(json_file_path, output_pdf_path=None, font_size=11, max_evidence_items=20):
    """
    Visualize hypotheses and their relationships from a JSON file.
    
    Parameters:
    -----------
    json_file_path : str
        Path to the JSON file containing idea data
    output_pdf_path : str, optional
        Path for the output PDF file. If None, will use the input filename with '_visualization.pdf'
    font_size : int, optional
        Font size for idea content (default: 11)
    max_evidence_items : int, optional
        Maximum number of evidence items to display (default: 20)
        
    Returns:
    --------
    str
        Path to the generated PDF file
    """
    
    # 設定預設輸出路徑
    if output_pdf_path is None:
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_pdf_path = f"{base_name}_visualization.pdf"
    
    # 載入 JSON 資料
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 建立 PDF 檔案
    pdf = PdfPages(output_pdf_path)
    
    # 輔助函數：取得想法詳情
    def get_idea_details(idea_id):
        for idea in data.get('ideas', []):
            if idea.get('id') == idea_id:
                return idea
        return None
    
    # 輔助函數：取得父節點 ID
    def get_parent_id(idea_id):
        idea = get_idea_details(idea_id)
        return idea.get('parent_id') if idea else None
    
    # 輔助函數：取得所有祖先節點
    def get_ancestors(idea_id):
        ancestors = []
        parent_id = get_parent_id(idea_id)
        while parent_id:
            ancestors.append(parent_id)
            parent_id = get_parent_id(parent_id)
        return ancestors
    
    # 輔助函數：文字換行
    def wrap_text(text, width=40):
        if not text:
            return ""
        return '\n'.join(textwrap.wrap(str(text), width=width))
    
    # 顏色調色盤（色盲友善）
    level_colors = [
        '#E69F00',  # Orange - 當前想法
        '#56B4E9',  # Blue - 第1層父節點
        '#009E73',  # Green - 第2層父節點
        '#F0E442',  # Yellow - 第3層父節點
        '#0072B2',  # Dark blue - 第4層父節點
        '#D55E00',  # Red-brown - 第5層父節點
        '#CC79A7',  # Pink - 第6層父節點
        '#999999',  # Grey - 第7層以上父節點
    ]
    
    # 取得頂級想法
    top_ideas = data.get('top_ideas', [])
    if not top_ideas:
        # 若未明確定義 top_ideas，則尋找 iteration > 0 的想法
        for idea in data.get('ideas', []):
            if idea.get('iteration', 0) > 0:
                top_ideas.append(idea['id'])
    
    print(f"Top Ideas: {top_ideas}")
    
    # 建立有向圖表示想法繼承關係
    G = nx.DiGraph()
    
    # 將所有想法加入圖中
    for idea in data.get('ideas', []):
        idea_id = idea.get('id')
        if not idea_id:
            continue
        
        parent_id = idea.get('parent_id')
        scores = idea.get('scores', {})
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        is_top = idea_id in top_ideas
        evidence_titles = [e.get('title', '') for e in idea.get('evidence', []) if isinstance(e, dict)]
        
        G.add_node(idea_id,
                   text=idea.get('text', ''),
                   scores=scores,
                   avg_score=avg_score,
                   is_top=is_top,
                   evidence=evidence_titles)
        
        # 加入父節點到子節點的邊
        if parent_id:
            G.add_edge(parent_id, idea_id)
    
    # 計算文字高度
    def calculate_text_height(text, width=60):
        if not text:
            return 1
        wrapped = wrap_text(text, width)
        return len(wrapped.split('\n'))
    
    # 為每個頂級想法建立視覺化
    for idea_id in top_ideas:
        # 取得所有相關節點（當前想法與其祖先）
        ancestors = get_ancestors(idea_id)
        relevant_nodes = [idea_id] + ancestors
        relevant_nodes_ordered = list(reversed(relevant_nodes))  # 最舊的祖先在前
        num_ideas = len(relevant_nodes)
        
        # 動態調整圖形大小
        base_height = 20
        height_per_idea = 2.5
        fig_height = max(base_height, 15 + (num_ideas * height_per_idea))
        
        # 建立圖形
        fig = plt.figure(figsize=(20, fig_height))
        
        # 建立網格配置
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 2],
                             height_ratios=[1, 3.5, 2.5],
                             left=0.05, right=0.95,
                             bottom=0.05, top=0.92,
                             wspace=0.1, hspace=0.3)
        
        # 建立節點到層級的映射（用於顏色）
        node_level_map = {node: idx for idx, node in enumerate(relevant_nodes_ordered)}
        node_color_map = {
            node: level_colors[min(idx, len(level_colors)-1)]
            for node, idx in node_level_map.items()
        }
        
        # 建立子圖
        subgraph = G.subgraph(relevant_nodes)
        
        # 1. 繼承關係圖（頂部區塊）
        ax1 = fig.add_subplot(gs[0, :])
        
        # 計算布局
        try:
            pos = nx.nx_agraph.graphviz_layout(
                subgraph, prog='dot',
                args='-Grankdir=TB -Gnodesep=2.0 -Granksep=1.5 -Gmargin=0.5'
            )
        except (ImportError, Exception) as e:
            print(f"Warning: Graphviz not available ({e}). Using spring layout.")
            avg_node_width = sum(len(node) * font_size * 0.7 for node in subgraph.nodes()) / max(len(subgraph.nodes()), 1)
            k_value = max(avg_node_width * 1.5, 5.0)
            pos = nx.spring_layout(subgraph, k=k_value, iterations=200, seed=42)
        
        # 計算節點矩形尺寸
        node_rects = {}
        for node in subgraph.nodes():
            x, y = pos[node]
            char_width_factor = 0.7
            base_width = len(node) * font_size * char_width_factor
            
            # 計算填充
            if len(node) > 40:
                padding_factor = 0.8
            elif len(node) > 30:
                padding_factor = 0.7
            else:
                padding_factor = 0.6
            
            padding = base_width * padding_factor
            rect_width = max(base_width + padding, font_size * 10)
            rect_height = 50
            
            node_rects[node] = {
                'x': x, 'y': y,
                'width': rect_width, 'height': rect_height,
                'color': node_color_map[node]
            }
        
        # 調整位置以避免重疊
        nodes_list = list(subgraph.nodes())
        adjusted_pos = {node: (node_rects[node]['x'], node_rects[node]['y']) for node in nodes_list}
        
        max_iterations = 15
        for iteration in range(max_iterations):
            overlaps_found = False
            for i, node1 in enumerate(nodes_list):
                x1, y1 = adjusted_pos[node1]
                w1, h1 = node_rects[node1]['width'], node_rects[node1]['height']
                
                for node2 in nodes_list[i+1:]:
                    x2, y2 = adjusted_pos[node2]
                    w2, h2 = node_rects[node2]['width'], node_rects[node2]['height']
                    
                    min_sep_x = (w1 + w2) / 2 + 30
                    min_sep_y = (h1 + h2) / 2 + 30
                    
                    if abs(x1 - x2) < min_sep_x and abs(y1 - y2) < min_sep_y:
                        overlaps_found = True
                        move_x = (min_sep_x - abs(x1 - x2)) / 2 + 10
                        move_y = (min_sep_y - abs(y1 - y2)) / 2 + 10
                        
                        if x1 < x2:
                            adjusted_pos[node1] = (x1 - move_x, adjusted_pos[node1][1])
                            adjusted_pos[node2] = (x2 + move_x, adjusted_pos[node2][1])
                        else:
                            adjusted_pos[node1] = (x1 + move_x, adjusted_pos[node1][1])
                            adjusted_pos[node2] = (x2 - move_x, adjusted_pos[node2][1])
                        
                        if y1 < y2:
                            adjusted_pos[node1] = (adjusted_pos[node1][0], y1 - move_y)
                            adjusted_pos[node2] = (adjusted_pos[node2][0], y2 + move_y)
                        else:
                            adjusted_pos[node1] = (adjusted_pos[node1][0], y1 + move_y)
                            adjusted_pos[node2] = (adjusted_pos[node2][0], y2 - move_y)
            
            if not overlaps_found:
                break
        
        # 繪製邊（先繪製，使其在節點後面）
        nx.draw_networkx_edges(subgraph, adjusted_pos, edge_color='gray',
                              arrows=True, arrowstyle='-|>', arrowsize=15,
                              width=1.5, ax=ax1)
        
        # 繪製節點矩形和文字
        for node in subgraph.nodes():
            x, y = adjusted_pos[node]
            rect_info = node_rects[node]
            
            # 繪製矩形
            rect = FancyBboxPatch(
                (x - rect_info['width']/2, y - rect_info['height']/2),
                rect_info['width'], rect_info['height'],
                boxstyle="round,pad=1.5",
                facecolor=rect_info['color'],
                edgecolor='black',
                alpha=0.8,
                linewidth=1.5
            )
            ax1.add_patch(rect)
            
            # 處理長 ID 的文字換行
            display_text = node
            if len(node) > 45:
                mid_point = len(node) // 2
                split_pos = node[:mid_point].rfind('_')
                if split_pos > 0:
                    display_text = node[:split_pos] + '\n' + node[split_pos+1:]
                else:
                    display_text = node[:mid_point] + '\n' + node[mid_point:]
            
            ax1.text(x, y, display_text, ha='center', va='center',
                    fontsize=font_size, fontweight='bold', color='black',
                    clip_on=True)
        
        # 設定軸範圍
        if adjusted_pos:
            x_values = [pos[0] for pos in adjusted_pos.values()]
            y_values = [pos[1] for pos in adjusted_pos.values()]
            
            max_rect_width = max(rect['width'] for rect in node_rects.values())
            x_padding = max(300, max_rect_width * 0.6)
            y_padding = 120
            
            ax1.set_xlim(min(x_values) - x_padding, max(x_values) + x_padding)
            ax1.set_ylim(min(y_values) - y_padding, max(y_values) + y_padding)
        
        ax1.set_title(f'Inheritance Relationship for Idea {idea_id}', fontsize=14, pad=10)
        ax1.axis('off')
        
        # 2. 想法內容（中間區塊）
        ax2 = fig.add_subplot(gs[1, :])
        ax2.axis('off')
        
        # 預先計算文字高度
        text_heights = {}
        for node in relevant_nodes_ordered:
            idea = get_idea_details(node)
            if idea:
                text = idea.get('text', '')
                text_height = calculate_text_height(text, width=60)
                
                # 檢查是否有方法細節
                is_current = node == idea_id
                method_extra_height = 0
                if is_current:
                    refined_method = idea.get('refined_method_details', {})
                    if refined_method:
                        method_desc = refined_method.get('description', '')
                        method_data = refined_method.get('method', {})
                        if method_desc:
                            method_extra_height += len(textwrap.wrap(method_desc, width=80)) + 3
                        if isinstance(method_data, dict) and 'System Architecture' in method_data:
                            arch_overview = method_data['System Architecture'].get('Overview', '')
                            if arch_overview:
                                method_extra_height += len(textwrap.wrap(arch_overview, width=80)) + 3
                
                scores = idea.get('scores', {})
                total_height = 3 + text_height + (2 if scores else 1) + method_extra_height
                text_heights[node] = total_height
        
        # 建立網格布局
        num_cols = min(2, num_ideas)
        num_rows = (num_ideas + num_cols - 1) // num_cols
        cell_width = 1.0 / num_cols
        cell_height = 1.0 / num_rows
        
        grid_positions = [(i // num_cols, i % num_cols) for i in range(num_ideas)]
        
        # 繪製想法內容框
        for i, node in enumerate(relevant_nodes_ordered):
            idea = get_idea_details(node)
            if not idea:
                continue
            
            idea_text = wrap_text(idea.get('text', ''), width=60)
            scores = idea.get('scores', {})
            score_text = ', '.join([f"{k}: {v:.1f}" for k, v in scores.items()]) if scores else "No scores"
            
            is_current = node == idea_id
            box_color = node_color_map[node]
            
            # 建立內容文字
            title = "CURRENT HYPOTHESIS: " if is_current else "PARENT HYPOTHESIS: "
            content = f"{title}{node}\n\nCONTENT: {idea_text}\n\nSCORES: {score_text}"
            
            # 加入方法細節（若為當前想法）
            refined_method = idea.get('refined_method_details', {})
            if refined_method and is_current:
                method_title = refined_method.get('title', '')
                method_desc = refined_method.get('description', '')
                method_name = refined_method.get('name', '')
                
                if method_name or method_title:
                    method_header = "\n\n=== METHOD DETAILS ===\n"
                    if method_name:
                        method_header += f"Name: {method_name}\n"
                    if method_title:
                        method_header += f"Title: {method_title}\n"
                    if method_desc:
                        desc_lines = textwrap.wrap(method_desc, width=80)
                        method_desc_short = '\n'.join(desc_lines[:5])
                        if len(desc_lines) > 5:
                            method_desc_short += "\n..."
                        method_header += f"Description: {method_desc_short}\n"
                    
                    method_data = refined_method.get('method', {})
                    if method_data and isinstance(method_data, dict):
                        if 'System Architecture' in method_data:
                            arch_overview = method_data['System Architecture'].get('Overview', '')
                            if arch_overview:
                                arch_lines = textwrap.wrap(arch_overview, width=80)
                                arch_short = '\n'.join(arch_lines[:3])
                                if len(arch_lines) > 3:
                                    arch_short += "\n..."
                                method_header += f"\nArchitecture Overview:\n{arch_short}\n"
                    
                    content += method_header
            
            # 計算位置
            row, col = grid_positions[i]
            x_pos = col * cell_width + 0.01
            y_pos = 1.0 - (row * cell_height) - 0.05
            box_width = cell_width * 0.98
            box_height = cell_height * 0.9
            
            # 調整字體大小
            content_length = len(content)
            adjusted_font_size = max(7, min(font_size, 11 - (content_length // 800)))
            
            props = dict(boxstyle='round,pad=0.8', facecolor=box_color, alpha=0.3)
            ax2.text(x_pos, y_pos, content, transform=ax2.transAxes,
                    fontsize=adjusted_font_size,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=props, wrap=True)
        
        ax2.set_title('Idea Content and Scores (Current and Parents)', fontsize=14, pad=10)
        
        # 3. 證據列表（底部區塊）
        ax3 = fig.add_subplot(gs[2, :])
        ax3.axis('off')
        
        # 收集證據
        all_evidence = []
        for node_id in relevant_nodes:
            idea = get_idea_details(node_id)
            if idea and 'evidence' in idea:
                for evidence in idea['evidence']:
                    if isinstance(evidence, dict):
                        evidence_item = {
                            'idea_id': node_id,
                            'title': evidence.get('title', ''),
                            'authors': evidence.get('authors', ''),
                            'year': evidence.get('year', ''),
                            'relevance_score': evidence.get('relevance_score', 0),
                            'color': node_color_map[node_id]
                        }
                        all_evidence.append(evidence_item)
        
        # 依相關性分數排序
        all_evidence.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # 建立表格資料
        table_data = []
        cell_colors = []
        
        for i, evidence in enumerate(all_evidence[:max_evidence_items]):
            row = [
                str(i+1),
                evidence['title'],
                evidence['idea_id'],
                str(evidence['year']),
                f"{evidence['relevance_score']:.2f}"
            ]
            table_data.append(row)
            cell_colors.append([evidence['color']] * 5)
        
        # 繪製表格
        if table_data:
            # 處理標題換行
            max_lines_per_title = 1
            wrapped_titles = []
            for row in table_data:
                title_text = row[1]
                existing_lines = title_text.count('\n') + 1
                
                if len(title_text) > 60:
                    parts = title_text.split('\n')
                    all_wrapped = []
                    for part in parts:
                        if len(part) > 60:
                            all_wrapped.extend(textwrap.wrap(part, width=60))
                        else:
                            all_wrapped.append(part)
                    wrapped_title = '\n'.join(all_wrapped)
                    wrapped_titles.append(wrapped_title)
                    max_lines_per_title = max(max_lines_per_title, len(all_wrapped))
                else:
                    wrapped_titles.append(title_text)
                    max_lines_per_title = max(max_lines_per_title, existing_lines)
            
            for i, wrapped_title in enumerate(wrapped_titles):
                table_data[i][1] = wrapped_title
            
            # 計算統一列高
            uniform_row_height = max(0.028 * max_lines_per_title + 0.012, 0.048)
            
            # 建立表格
            table = plt.table(
                cellText=table_data,
                colLabels=['#', 'Title', 'Idea', 'Year', 'Relevance'],
                cellLoc='left',
                loc='center',
                colWidths=[0.02, 0.70, 0.15, 0.04, 0.09]
            )
            
            # 套用顏色
            for (i, j), cell in table.get_celld().items():
                if i > 0:
                    cell.set_facecolor(mcolors.to_rgba(cell_colors[i-1][j], alpha=0.2))
            
            # 格式化表格
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            
            # 設定統一列高
            for i in range(len(table_data) + 1):
                if i == 0:
                    for j in range(5):
                        cell = table[(i, j)]
                        cell.set_height(0.045)
                        cell.get_text().set_fontsize(7)
                        cell.get_text().set_weight('bold')
                        cell.get_text().set_va('center')
                else:
                    for j in range(5):
                        cell = table[(i, j)]
                        cell.set_height(uniform_row_height)
            
            # 設定文字屬性
            for i in range(1, len(table_data) + 1):
                for j in range(5):
                    cell = table[(i, j)]
                    if j == 1:  # Title column
                        cell.get_text().set_fontsize(7)
                        cell.get_text().set_va('top')
                        cell.get_text().set_ha('left')
                    elif j == 2:  # Idea column
                        cell.get_text().set_fontsize(6)
                        cell.get_text().set_wrap(True)
                        cell.get_text().set_va('top')
                        cell.get_text().set_ha('left')
                    else:
                        cell.get_text().set_fontsize(7)
                        cell.get_text().set_va('top')
                        cell.get_text().set_ha('left')
        
        ax3.set_title('Evidence Used by This Idea and Its Ancestors', fontsize=14, pad=45, y=1.0)
        
        # 設定整體標題
        plt.suptitle(f'Comprehensive View of Top Idea: {idea_id}', fontsize=16, y=0.98)
        
        # 儲存圖形
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # 關閉 PDF
    pdf.close()
    print(f"Visualization saved to {output_pdf_path}")
    
    return output_pdf_path
