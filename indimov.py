import os
import pandas as pd
import matplotlib.pyplot as plt
import ast

# Path to folder
folder = "Individual_movies"

def parse_te(val):
    lst = ast.literal_eval(val)
    return lst[0]

def parse_p(val):
    d = ast.literal_eval(val)
    return d[0]

def format_sig(p):
    if p < 0.001:
        return ("***", "red", 90)
    elif p < 0.01:
        return ("**", "red", 90)
    elif p < 0.05:
        return ("*", "red", 90)
    else:
        return ("n.s.", "black", 0)

# Movie intervals and durations
movie_intervals = {
    "inception": (20, 246),
    "facebook": (267, 525),
    "ocean11": (545, 794),
    "last": (815, 897)  # replace 'last' with 4th movie name
}

def movie_label(movie):
    start, end = movie_intervals[movie]
    duration = end - start
    return f"{movie.capitalize()} ({duration} seconds)"

# Read CSVs
data = []
for file in os.listdir(folder):
    if file.endswith(".csv"):
        movie, roi, input_, _ = file.replace(".csv","").split("_", 3)
        df = pd.read_csv(os.path.join(folder, file))
        row = {
            "movie": movie,
            "roi": roi,
            "input": input_,
            "te_acw_forward": parse_te(df["te_acw_forward"].values[0]),
            "te_raw_forward": parse_te(df["te_raw_forward"].values[0]),
            "p_af_mbb": parse_p(df["p_af_mbb"].values[0]),
            "p_rf_mbb": parse_p(df["p_rf_mbb"].values[0])
        }
        data.append(row)

df_all = pd.DataFrame(data)

# Movie order
movies = ["inception", "facebook", "ocean11", "last"]
rois = ["A1", "TA2", "PSL"]
roi_colors = ["blue", "green", "red"]
inputs = sorted(df_all["input"].unique())

# Create figure
fig, axes = plt.subplots(4, len(movies), figsize=(16,12), sharey=False)  # no shared y

# Fixed Y range and annotation height
ymax = 5
sig_y = 4.0  # height for stars/n.s.

for col, movie in enumerate(movies):
    for row_idx, input_ in enumerate(inputs):
        r_acw = 0 if row_idx == 0 else 2
        r_raw = 1 if row_idx == 0 else 3
        subset = df_all[(df_all["movie"] == movie) & (df_all["input"] == input_)]
        
        # TE ACW
        ax1 = axes[r_acw, col]
        vals = [subset[subset["roi"] == roi]["te_acw_forward"].values[0] for roi in rois]
        ps = [subset[subset["roi"] == roi]["p_af_mbb"].values[0] for roi in rois]
        bars = ax1.bar(rois, vals, color=roi_colors, width=0.4)
        for i, (v, p) in enumerate(zip(vals, ps)):
            txt, color, rot = format_sig(p)
            ax1.text(i, sig_y, txt, ha='center', va='bottom', rotation=rot, color=color, fontsize=12, fontweight='bold')
        ax1.set_ylim(0, ymax)
        ax1.set_ylabel("TE (Dynamic ACW)", fontweight='bold')
        ax1.set_title(movie_label(movie), fontweight='bold')

        # TE RAW
        ax2 = axes[r_raw, col]
        vals = [subset[subset["roi"] == roi]["te_raw_forward"].values[0] for roi in rois]
        ps = [subset[subset["roi"] == roi]["p_rf_mbb"].values[0] for roi in rois]
        bars = ax2.bar(rois, vals, color=roi_colors, width=0.4)
        for i, (v, p) in enumerate(zip(vals, ps)):
            txt, color, rot = format_sig(p)
            ax2.text(i, sig_y, txt, ha='center', va='bottom', rotation=rot, color=color, fontsize=12, fontweight='bold')
        ax2.set_ylim(0, ymax)
        ax2.set_ylabel("TE (BOLD signal)", fontweight='bold')
        ax2.set_title(movie_label(movie), fontweight='bold')

# Headings for input sections
fig.text(0.5, 0.98, "Input to Brain Transfer Entropy - Sentence similarity", ha='center', fontsize=16, fontweight='bold')
fig.text(0.5, 0.48, "Input to Brain Transfer Entropy - Word Depth", ha='center', fontsize=16, fontweight='bold')

# Adjust vertical spacing
fig.subplots_adjust(hspace=0.6, wspace=0.3)

# Bold tick labels for all subplots
for ax_row in axes:
    for ax in ax_row:
        ax.tick_params(axis='both', labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
# After plotting everything, before saving/showing:
plt.tight_layout(rect=[0, 0, 1, 0.95])  # adjust top to leave space for your headings
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.6, wspace=0.3)

plt.savefig("transfer_entropy_individual_movies.png", dpi=600)
plt.show()
