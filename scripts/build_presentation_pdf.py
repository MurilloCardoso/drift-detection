"""Gera slides PDF a partir das evidências verificadas do pipeline Statlog."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "apresentacao_statlog_star.pdf"
TMP = ROOT / "tmp" / "pdfs"
W, H = 960, 540
INK, MUTED, BLUE = HexColor("#142039"), HexColor("#52627A"), HexColor("#1463C8")


def font_setup() -> None:
    base = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("Arial", str(base / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(base / "arialbd.ttf")))


def line(
    c: canvas.Canvas, text: str, x: float, y: float, size: int = 20, bold: bool = False, color=INK
) -> None:
    c.setFont("Arial-Bold" if bold else "Arial", size)
    c.setFillColor(color)
    c.drawString(x, y, text)


def wrap(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    size: int = 20,
    leading: int = 28,
    color=INK,
) -> float:
    words, row = text.split(), ""
    for word in words:
        candidate = f"{row} {word}".strip()
        if pdfmetrics.stringWidth(candidate, "Arial", size) > width and row:
            line(c, row, x, y, size, color=color)
            y -= leading
            row = word
        else:
            row = candidate
    if row:
        line(c, row, x, y, size, color=color)
        y -= leading
    return y


def base(c: canvas.Canvas, number: int, title: str, section: str) -> None:
    c.setFillColor(HexColor("#FFFFFF"))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    line(c, section.upper(), 44, 502, 13, True, BLUE)
    line(c, title, 44, 455, 30, True)
    c.setStrokeColor(HexColor("#D5E0EC"))
    c.line(44, 433, 916, 433)
    line(c, f"{number:02d} / 08", 858, 24, 12, color=MUTED)


def note(c: canvas.Canvas, text: str) -> None:
    """O roteiro é entregue à parte, mantendo os slides limpos."""


def screenshot(
    c: canvas.Canvas,
    source: str,
    x: int,
    y: int,
    width: int,
    height: int,
    crop: tuple[int, int, int, int],
) -> None:
    with Image.open(TMP / source) as image:
        clipped = image.crop(crop)
        target = TMP / f"slide_{source}"
        clipped.save(target)
    c.drawImage(str(target), x, y, width, height, preserveAspectRatio=True, anchor="c")


def main() -> None:
    font_setup()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(
        (ROOT / "artifacts/reports/statlog_baseline_metrics.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (ROOT / "artifacts/reports/statlog_drift_summary.json").read_text(encoding="utf-8")
    )
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    c.setTitle("Tech Challenge Fase 4 - Statlog German Credit")

    c.setFillColor(HexColor("#F7F9FC"))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    cover_image = TMP / "cover_dashboard.png"
    with Image.open(TMP / "dashboard.png") as image:
        image.crop((210, 30, 990, 1000)).save(cover_image)
    c.drawImage(str(cover_image), 535, 0, 425, 540)
    line(c, "TECH CHALLENGE", 50, 470, 16, True, BLUE)
    line(c, "FASE 4  /  MLET", 50, 445, 14, True, MUTED)
    line(c, "Crédito sob", 50, 330, 43, True)
    line(c, "monitoramento", 50, 273, 43, True)
    wrap(c, "Contrato de dados, classificador de risco e detecção de drift", 50, 202, 430, 23, 30)
    line(c, "Statlog German Credit", 50, 70, 16, True, BLUE)
    line(c, "1.000 casos  ·  20 atributos  ·  alvo real", 50, 45, 13, color=MUTED)
    note(
        c,
        "0:00-0:25 | Apresente o problema: um modelo de crédito pode receber dados ruins "
        "ou mudar de comportamento sem aviso.",
    )
    c.showPage()

    base(c, 2, "Situation e Task", "Problema e objetivo")
    y = 378
    for text in [
        "Risco: degradação silenciosa em decisões de crédito.",
        "Objetivo: barrar dados inválidos, medir qualidade do classificador e detectar drift.",
        "Base: 700 casos de bom risco e 300 de mau risco. Classe positiva = mau risco (2).",
    ]:
        y = wrap(c, text, 48, y, 860, 22, 30) - 22
    note(
        c,
        "0:25-0:55 | Explique o cenário da fintech e por que um alerta estatístico "
        "deve levar à investigação, não a uma decisão automática.",
    )
    c.showPage()

    base(c, 3, "Contrato bloqueia o lote inválido", "Action - qualidade")
    line(c, "1.000", 50, 344, 54, True, BLUE)
    line(c, "registros aprovados", 50, 306, 19)
    line(c, "14", 50, 223, 54, True, BLUE)
    line(c, "violações bloqueadas", 50, 185, 19)
    wrap(
        c, "Pandera valida tipos, faixas, códigos, alvo, unicidade e duplicatas.", 460, 335, 420, 22
    )
    note(
        c,
        "0:55-1:35 | Execute demo. Mostre o contrato aprovado e o arquivo "
        "statlog_invalid_failure_cases.csv com falhas de idade, categoria, duração e alvo.",
    )
    c.showPage()

    base(c, 4, "Classificador com teste separado", "Action - modelo")
    line(c, f"{metrics['train_rows']} treino / {metrics['test_rows']} teste", 48, 365, 27, True)
    line(c, f"ROC AUC  {metrics['roc_auc']:.3f}", 48, 284, 31, True, BLUE)
    line(c, f"F1  {metrics['f1']:.3f}", 48, 231, 31, True, BLUE)
    wrap(
        c,
        "Regressão logística com divisão estratificada. O modelo é ajustado "
        "apenas no treino; as métricas são calculadas no teste.",
        485,
        359,
        420,
        20,
    )
    note(
        c,
        "1:35-2:15 | Aponte a classe positiva e a separação treino/teste. "
        "Essas métricas são acadêmicas, não autorização para decisão real.",
    )
    c.showPage()

    base(c, 5, "Evidently detecta o drift", "Action - distribuição")
    screenshot(c, "drift.png", 40, 105, 880, 315, (0, 100, 1600, 670))
    note(
        c,
        "2:15-3:00 | Mostre o print real: age e credit_amount sofreram deslocamento. "
        "O alerta global do Evidently está ativo; a simulação é sintética.",
    )
    c.showPage()

    base(c, 6, "Monitoramento operacional", "Result - observabilidade")
    screenshot(c, "dashboard.png", 40, 105, 880, 315, (0, 25, 1600, 600))
    note(
        c,
        "3:00-3:40 | Mostre status, latência, volume e alertas. "
        "Há {count} alertas no resumo PSI/KS, incluindo a probabilidade prevista.".format(
            count=len(summary["drifted_columns"])
        ),
    )
    c.showPage()

    base(c, 7, "Governança e limites da evidência", "Result - interpretação")
    y = 380
    for text in [
        "LGPD: finalidade, acesso restrito, retenção definida e direitos do titular.",
        "Revisar vieses e manter avaliação humana antes de qualquer uso real.",
        "O experimento demonstra data drift; não demonstra concept drift "
        "ou perda real de desempenho.",
    ]:
        y = wrap(c, text, 48, y, 865, 22, 30) - 22
    note(
        c,
        "3:40-4:25 | Explique a análise causal limitada: alteramos duas variáveis "
        "e observamos mudança estatística. Não há desfechos futuros de produção.",
    )
    c.showPage()

    base(c, 8, "Fechamento e entrega", "Result - próximos passos")
    y = 380
    for text in [
        "Validado localmente: 13 testes, Ruff e comandos demo, drift e monitor.",
        "Pendente para entrega: commit do dataset extraído e alterações, "
        "CI no GitHub e link do vídeo STAR.",
        "Em produção: acompanhar desfechos reais, desempenho por grupo "
        "e infraestrutura do serviço.",
    ]:
        y = wrap(c, text, 48, y, 865, 22, 30) - 22
    note(
        c,
        "4:25-5:00 | Encerre com a lição: monitorar dados, modelo e operação juntos. "
        "Confirme os itens pendentes antes de submeter.",
    )
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
