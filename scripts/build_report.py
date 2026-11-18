"""Build the consolidated Markdown/PDF from saved evidence (requires reportlab)."""
import csv
import json
from pathlib import Path
from statistics import mean
from html import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
REPORTS = ROOT / 'reports'


def rows(name):
    with (OUT / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    benchmark = rows('model_experimentation_results.csv')
    cv = rows('grouped_cv_folds.csv')
    tuning = rows('bayesian_tuning_comparison.csv')
    route = rows('route_reconstruction_continuity_comparison.csv')
    evaluator = json.loads((OUT / 'evaluator/summary.json').read_text())
    history = json.loads((OUT / 'bayesian_search_history.json').read_text())
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SmallCell', fontName='Helvetica', fontSize=8, leading=10))
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 14
    styles['BodyText'].spaceAfter = 8
    story, markdown = [], []

    def heading(text, level=1):
        markdown.append('#' * level + ' ' + text + '\n')
        story.append(Paragraph(escape(text), styles['Title' if level == 1 else 'Heading2']))

    def paragraph(text):
        markdown.append(text + '\n')
        story.append(Paragraph(escape(text), styles['BodyText']))

    def table(headers, data, widths=None):
        markdown.append('| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |')
        markdown.extend('| ' + ' | '.join(map(str, r)) + ' |' for r in data)
        markdown.append('')
        cells = [[Paragraph(escape(str(v)), styles['SmallCell']) for v in r] for r in [headers] + data]
        t = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbeafe')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f1f5f9')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 7), ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#94a3b8'))]))
        story.extend([t, Spacer(1, 12)])

    def percent(v):
        return f'{100 * float(v):.2f}%'

    heading('PathSnap | Technical Report')
    paragraph('Verified integration - 18 November 2026. GPS trajectory reconstruction with geometric candidate features, LightGBM and HMM/Viterbi. This report is generated from saved results rather than manually copied UI figures.')
    heading('1. Scope and data separation', 2)
    paragraph('The data contains 80 training trips, 10 student-test trips and 10 reserved evaluator trips. Each trip supplies GPS observations, nodes, directed arcs and reference arc labels. Evaluator data is used only for final scoring after prediction; it is excluded from fitting, feature-label alignment, model selection and tuning. The evaluator API does not read reference labels, returns no evaluator F1, and renders no reference geometry.')
    heading('2. Preprocessing and inference', 2)
    paragraph('Longitude and latitude are projected to a local metric coordinate system. A 50 m spatial grid searches a nominal 40 m radius, retaining up to eight candidate arcs and a minimum of three when available. Thirteen geometric and motion features feed a binary match classifier. Training labels use an approximate moving-window projection to the reference route. Candidate-label quality therefore depends on this alignment.')
    paragraph('Classifier probabilities and spatial likelihood form HMM emissions with sigma_z=4.07 m, beta=3.0 and alpha_ml=0.70. Viterbi selects observation arcs; directed shortest paths insert intermediate arcs. Missing candidates produce an explicit error. If selected arcs cannot be connected, the decoder records the gap; the API returns partial status, connected route sections and measured continuity. It never draws an artificial connection across these sections.')
    heading('3. Metric definitions', 2)
    paragraph('Candidate accuracy, precision, recall and binary F1 classify candidate pairs. Macro F1 averages F1 for both classes equally. Observation top-1 accuracy selects the highest-scored candidate per (trip_id, obs_id), including candidate misses in its denominator. Arc-set precision/recall/F1 compare unique route arc sets: they ignore traversal order and repeated traversals. Legacy seq_f1 column names retain this arc-set meaning. Observation continuity concerns consecutive matched GPS arcs; full-route continuity includes inserted bridge arcs.')
    story.append(PageBreak())
    heading('4. Five-model benchmark and grouped CV', 2)
    paragraph('Three-fold GroupKFold uses all 80 training trips for each model. Fold-specific scalers are fitted only on training folds. No trip appears in both sides of a fold. KNN fitting uses a deterministic 15,000-row training subsample per fold; validation folds are complete. MLP uses a fixed 40-epoch budget, without row-level early-stopping validation. It reached the epoch limit and emitted a convergence warning; its scores describe that budget-limited fit, not a fully converged network. Full fold trip manifests and all metrics are saved in grouped_cv_folds.csv.')
    table(['Model', 'Test top-1', 'Test macro F1', 'CV macro F1 mean', 'CV std'],
          [[r['model'], percent(r['top1_accuracy']), percent(r['macro_f1']), percent(r['cv_macro_f1_mean']), percent(r['cv_macro_f1_std'])] for r in benchmark], [150, 80, 85, 95, 65])
    paragraph(f'{len(cv)} fold results are retained. Student-test scores are from 10 trips; grouped CV scores are from training trips only. Timing and training-budget differences are documented rather than treated as equal-cost comparisons.')
    heading('5. Bayesian tuning and final artifact', 2)
    paragraph(f'A seeded TPE search completed {len(history)} saved trials with three-fold grouped CV over a deterministic subset of 40 training trips. The objective is mean observation top-1 accuracy. Fold scores, full parameters, trial state and timestamps are saved in bayesian_search_history.json/CSV, with resumable SQLite state. The original unsaved search history was unavailable; this is a fresh, reproducible search. The selected model was refitted on all 80 training trips in the documented runtime.')
    table(['Model version', 'Top-1', 'Macro F1', 'Classifier continuity', 'Arc-set F1'],
          [[r['version'], percent(r['top1_accuracy']), percent(r['macro_f1']), percent(r['route_continuity']), percent(r['seq_f1'])] for r in tuning], [175, 65, 70, 90, 75])
    paragraph('Selection uses training CV rather than evaluator performance. Test-set improvement is not guaranteed for every metric. Final model SHA-256: ' + evaluator['model_sha256'])
    story.append(PageBreak())
    heading('6. Route reconstruction on student-test trips', 2)
    table(['Trip', 'GPS points', 'Raw continuity', 'HMM obs continuity', 'Full continuity', 'Arc-set F1'],
          [[r['trip_id'], r['num_gps'], percent(r['continuity_raw']), percent(r['continuity_hmm']), percent(r['continuity_full_route']), percent(r['seq_f1_hmm'])] for r in route], [70, 65, 80, 95, 80, 85])
    paragraph('Mean classifier continuity: ' + percent(mean(float(r['continuity_raw']) for r in route)) +
              '; HMM observation continuity: ' + percent(mean(float(r['continuity_hmm']) for r in route)) +
              '; HMM arc-set F1: ' + percent(mean(float(r['seq_f1_hmm']) for r in route)) + '.')
    paragraph(f"The decoded student-test routes include {sum(int(r['disconnected_transition_count']) for r in route)} unbridgeable transitions. These are reported explicitly and explain why full-route continuity is not universally 100%. Existing claims of mathematically guaranteed continuity have been removed.")
    heading('7. Reserved evaluator results', 2)
    table(['Measure', 'Result'], [
        ['Trips / GPS observations', f"{evaluator['trips']} / {evaluator['gps_points']:,}"],
        ['Mean classifier-only arc-set F1', percent(evaluator['mean_raw_arc_set_f1'])],
        ['Mean HMM route arc-set F1', percent(evaluator['mean_arc_set_f1'])],
        ['Mean observation continuity', percent(evaluator['mean_observation_continuity'])],
        ['Mean full-route continuity', percent(evaluator['mean_full_route_continuity'])],
        ['Total measured inference time', f"{evaluator['total_inference_seconds']:.2f} seconds"]], [290, 185])
    paragraph('Scores are unweighted means over trips. All ten predicted route files use one zero-based arc ID per line and are stored separately from reference files. Labels are read only after predictions are persisted. The weakest evaluator trip should be retained as an honest limitation rather than used for further tuning.')
    story.append(PageBreak())
    heading('8. Runtime, tests and reproducibility', 2)
    paragraph('Verified runtime: Python 3.12, NumPy 1.26.4, scikit-learn 1.4.2 and LightGBM 4.6.0. requirements.txt pins the remaining direct dependencies. The original saved model reported scikit-learn 1.9.0 and crashed with the local LightGBM 4.7.0 runtime. The new artifact was trained and tested locally, removing that serialization mismatch. Thread counts are limited to two for repeatability and predictable laptop load.')
    for evidence in ['review_tests.txt', 'end_to_end_verification.txt']:
        text = (OUT / evidence).read_text(encoding='utf-8')
        summary = next((line for line in reversed(text.splitlines()) if 'passed in ' in line or 'TOTAL TESTS:' in line), 'See evidence file')
        paragraph(evidence + ': ' + summary.strip())
    paragraph('Regression coverage includes grouping/macro F1, label-free loading, reference-free map HTML, disconnected/empty candidates, evaluator listing, invalid API parameters and payloads, old-map isolation and live blind inference. End-to-end verification covers loading, geometry, features, model probabilities, decoder runtime, metrics and both inference endpoints. Tests exit nonzero on failure; timings are hardware dependent.')
    heading('9. Deployment and two-laptop demonstration', 2)
    paragraph('On a new laptop: create a Python 3.12 virtual environment, install requirements.txt, transfer raw data separately, then run run_project.ps1 test followed by run_project.ps1 serve. The launcher prefers .venv, falling back to the verified local overlay on this laptop. PATHSNAP_DATA_DIR can point at an external directory containing train, student_test and evaluator_test.')
    paragraph('For a two-laptop demo, serve from laptop A and open its LAN address on laptop B on a trusted network. Follow local firewall policy. There is no authentication, so stop the server after demonstration. Map scripts, styles and tiles require internet access. The service mounts only public prediction maps; legacy offline EDA maps are not exposed.')
    heading('10. Submission contents and limitations', 2)
    paragraph('Keep src, api, tests, requirements.txt, README.md, run_project.ps1, reports, model artifacts and result evidence in one shared repository. Exclude raw data, dataset caches, dynamic maps, runtimes and Python caches using .gitignore. Each contributor should review and commit their own changes; no identities, commits or pushes are fabricated. No external submission destination or additional marking rubric was provided.')
    paragraph('Remaining scientific limitations: approximate label alignment, arc-set rather than order-sensitive route scoring, KNN training subsampling, a fixed MLP epoch budget and incomplete connectivity in some supplied networks. These are stated explicitly rather than hidden behind aggregate scores.')
    story.append(PageBreak())
    heading('Saved comparison figures', 2)
    for name, caption in [('metric_comparison_5_models.png', 'Candidate metrics and observation top-1 across five model families.'),
                          ('route_continuity_classifier_vs_hmm.png', 'Student-test observation continuity and route arc-set F1.')]:
        file = OUT / 'figures' / name
        paragraph(caption)
        w, h = ImageReader(str(file)).getSize()
        story.append(Image(str(file), width=475, height=h * 475 / w))
        story.append(Spacer(1, 14))
        markdown.append(f'![{caption}](../outputs/figures/{name})\n')
    for name, caption in [('student_live.jpg', 'Student-test live prediction: measured continuity and correctly labelled arc-set F1.'),
                          ('evaluator_blind.jpg', 'Reserved evaluator prediction: F1 is unavailable and no reference layer is present.')]:
        file = REPORTS / 'screenshots' / name
        if file.exists():
            story.append(PageBreak())
            heading('Demonstration capture', 2)
            paragraph(caption + ' Genuine browser screenshot captured from the running local API.')
            w, h = ImageReader(str(file)).getSize()
            scale = min(475 / w, 600 / h)
            story.append(Image(str(file), width=w * scale, height=h * scale))
            markdown.append(f'![{caption}](screenshots/{name})\n')

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#cbd5e1'))
        canvas.line(48, 40, 547, 40)
        canvas.setFont('Helvetica', 8)
        canvas.drawString(48, 27, 'PathSnap | Verified technical report | 18 November 2026')
        canvas.drawRightString(547, 27, str(doc.page))

    (REPORTS / 'PathSnap_Technical_Report.md').write_text('\n'.join(markdown), encoding='utf-8')
    SimpleDocTemplate(str(REPORTS / 'PathSnap_Technical_Report.pdf'), pagesize=(595, 842),
                      leftMargin=60, rightMargin=60, topMargin=48, bottomMargin=58,
                      title='PathSnap Technical Report').build(story, onFirstPage=footer, onLaterPages=footer)
    print('Consolidated Markdown and PDF written.')


if __name__ == '__main__':
    main()
