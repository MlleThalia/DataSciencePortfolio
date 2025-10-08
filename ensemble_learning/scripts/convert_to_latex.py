import subprocess
from pathlib import Path
import os

##################################################################
###########  Change dataframe cell format

def format_result(cell, show_test=False):
    """
    Formate une cellule contenant (test_acc, mean_acc, std_acc)
    en une chaîne 'mean±std' (ou inclut test_acc si show_test=True).
    """
    test, mean, std = cell
    if show_test:
        return f"{test:.1f}/{mean:.1f}±{std:.1f}"
    else:
        return f"{mean:.1f}±{std:.1f}"

##################################################################
########### Convert dataframe to a latex table
    
def results_to_latex(df, caption="Model comparison", label="tab:results", show_test=False):
    df_latex = df.copy()
    for col in df_latex.columns[1:]:
        df_latex[col] = df_latex[col].apply(lambda x: format_result(x, show_test))
        df_latex[col] = df_latex[col].str.replace("±", "$\\pm$")
    return df_latex.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c"*(df_latex.shape[1]-1),
        caption=caption,
        label=label
    )

##################################################################
########### Generate .pdf file from latex template

def inject_table_in_template(latex_table, template_path="template.tex", output_pdf="results.pdf"):
    """
    Injecte un tableau LaTeX dans un template existant et compile le tout en PDF.
    """
    # Charger le template
    template_text = Path(template_path).read_text(encoding="utf-8")

    # Remplacer le placeholder
    filled_text = template_text.replace("%TABLE_PLACEHOLDER%", latex_table)

    # Sauvegarder le fichier temporaire
    tex_file = Path("filled_template.tex")
    tex_file.write_text(filled_text, encoding="utf-8")

    # Compiler avec pdflatex (MiKTeX, TeXLive, etc.)
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(tex_file)],
            check=True
        )
        print(f"✅ Compilation réussie → {output_pdf}")
    except subprocess.CalledProcessError as e:
        print("❌ Erreur LaTeX :", e)
    
     # Renommer le PDF généré
    generated_pdf = tex_file.with_suffix(".pdf")
    if generated_pdf.exists():
        generated_pdf.rename(output_pdf)
    else:
        print("⚠️ La compilation s’est terminée sans erreur mais le PDF n’a pas été trouvé.")

    # Nettoyer les fichiers auxiliaires
    for ext in [".aux", ".log", ".out"]:
        f = tex_file.with_suffix(ext)
        if f.exists():
            f.unlink()

##################################################################
########### Remove .pdf file

def remove_pdf_if_exists(filepath):
    """
    Supprime le fichier PDF s'il existe.
    
    Paramètres
    ----------
    filepath : str
        Chemin complet ou relatif du fichier PDF.
    """
    if os.path.isfile(filepath):
        os.remove(filepath)
        print(f"✅ Fichier supprimé : {filepath}")
    else:
        print(f"ℹ️ Aucun fichier trouvé à cet emplacement : {filepath}")