import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'api.dart';
import 'browser_inspector.dart';
import 'judgment.dart';

const _api = String.fromEnvironment('HANCUT_API');

/// 판독 화면 — X-ray 사진을 고르면 판정하고, 사진 판정과 12종 품목별 구간을 보여준다.
/// 품목을 누르면 사진 위에 그 품목이 가장 의심되는 자리를 표시한다. 재검은 반드시 판독관이 본다.
///
/// 판정 방법: `--dart-define=HANCUT_API=…` 를 주면 판정 서버, 웹에서 주지 않으면 브라우저 안(ONNX).
class HomePage extends StatefulWidget {
  const HomePage({super.key, this.pickImage, this.inspector});

  /// 시험에서 사진 고르기와 판정 방법을 바꿔 끼우기 위한 자리.
  final Future<(Uint8List, String)?> Function()? pickImage;
  final Inspector? inspector;

  @override
  State<HomePage> createState() => _HomePageState();
}

enum _ModelState { loading, ready, failed }

class _HomePageState extends State<HomePage> {
  final _server = TextEditingController(text: _api.isEmpty ? 'http://localhost:8000' : _api);
  late final bool _inBrowser = widget.inspector == null && _api.isEmpty && browserInspectionAvailable;
  late final Inspector? _browser = _inBrowser ? BrowserInspector() : null;
  _ModelState _model = _ModelState.ready;
  Uint8List? _image;
  String? _name;
  Inspection? _result;
  String? _selected;
  String? _error;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    if (_browser != null) {
      _model = _ModelState.loading;
      _browser.warmup().then(
        (_) {
          if (mounted) setState(() => _model = _ModelState.ready);
        },
        onError: (Object e) {
          if (mounted) setState(() => _model = _ModelState.failed);
        },
      );
    }
  }

  @override
  void dispose() {
    _server.dispose();
    super.dispose();
  }

  Inspector get _inspector => widget.inspector ?? _browser ?? InspectionClient(_server.text.trim());

  Future<(Uint8List, String)?> _pick() async {
    if (widget.pickImage != null) return widget.pickImage!();
    final file = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (file == null) return null;
    return (await file.readAsBytes(), file.name);
  }

  Future<void> _inspect() async {
    final picked = await _pick();
    if (picked == null) return;
    final (bytes, name) = picked;
    setState(() {
      _image = bytes;
      _name = name;
      _result = null;
      _selected = null;
      _error = null;
      _busy = true;
    });
    try {
      final result = await _inspector.inspect(bytes, filename: name.split('.').first);
      if (mounted) {
        setState(() {
          _result = result;
          _selected = result.sorted.first.item;
          _model = _ModelState.ready;
        });
      }
    } on InspectionError catch (e) {
      if (mounted) setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Widget _engine(BuildContext context) {
    final t = Theme.of(context).textTheme;
    if (!_inBrowser) {
      return TextField(
        controller: _server,
        decoration: const InputDecoration(labelText: '판정 서버 주소', border: OutlineInputBorder(), isDense: true),
      );
    }
    final (icon, text) = switch (_model) {
      _ModelState.loading => (Icons.downloading, '판정 모델 불러오는 중… (약 38MB, 처음 한 번)'),
      _ModelState.ready => (Icons.lock_outline, '이 브라우저에서 판정해요 — 사진은 밖으로 나가지 않아요'),
      _ModelState.failed => (Icons.error_outline, '판정 모델을 불러오지 못했어요. 새로고침해 주세요.'),
    };
    return Row(
      children: [
        Icon(icon, size: 18),
        const SizedBox(width: 8),
        Expanded(child: Text(text, style: t.bodySmall)),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final wide = MediaQuery.sizeOf(context).width >= 900;
    final result = _result;
    final selected = result?.items.where((e) => e.item == _selected).firstOrNull;
    final left = <Widget>[
      _Intro(),
      const SizedBox(height: 16),
      _engine(context),
      const SizedBox(height: 12),
      FilledButton.icon(
        onPressed: _busy ? null : _inspect,
        icon: const Icon(Icons.upload_file),
        label: Text(_busy ? '판정 중…' : 'X-ray 사진 올려서 판정하기'),
      ),
      if (_image != null) ...[
        const SizedBox(height: 16),
        _XrayView(image: _image!, size: result?.imageSize, item: selected),
        const SizedBox(height: 6),
        Text(
          selected?.box == null
              ? (_name ?? '')
              : '${selected!.label} — 이 사진에서 ${selected.label} 점수가 가장 높은 자리 (${selected.zone.label})'
                    '${selected.zone == JudgmentZone.autoClear ? ' · 점수가 기준 아래라 물건이 아닐 가능성이 커요' : ''}',
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    ];
    final right = <Widget>[
      if (_error != null) _ErrorCard(message: _error!),
      if (result != null)
        _ResultView(result: result, selected: _selected, onSelect: (item) => setState(() => _selected = item)),
      if (result == null && _error == null) const _ZoneGuide(),
    ];
    return Scaffold(
      appBar: AppBar(title: const Text('한컷점검 · X-ray 판독 보조')),
      body: wide
          ? Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SizedBox(
                  width: 460,
                  child: ListView(padding: const EdgeInsets.all(20), children: left),
                ),
                const VerticalDivider(width: 1),
                Expanded(
                  child: ListView(padding: const EdgeInsets.all(20), children: right),
                ),
              ],
            )
          : ListView(padding: const EdgeInsets.all(16), children: [...left, const SizedBox(height: 20), ...right]),
    );
  }
}

/// 사진과, 고른 품목의 '가장 의심되는 자리' 박스.
class _XrayView extends StatelessWidget {
  const _XrayView({required this.image, required this.size, required this.item});

  final Uint8List image;
  final Size? size;
  final ItemJudgment? item;

  @override
  Widget build(BuildContext context) {
    final picture = Image.memory(
      image,
      fit: size == null ? BoxFit.contain : BoxFit.fill,
      gaplessPlayback: true,
      errorBuilder: (_, _, _) => const SizedBox(height: 80, child: Center(child: Text('미리보기를 그릴 수 없는 사진이에요'))),
    );
    final box = item?.box;
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: size == null
          ? picture
          : AspectRatio(
              aspectRatio: size!.aspectRatio,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  picture,
                  if (box != null)
                    CustomPaint(
                      key: const Key('box'),
                      painter: _BoxPainter(box: box, image: size!, color: item!.zone.color, label: item!.label),
                    ),
                ],
              ),
            ),
    );
  }
}

class _BoxPainter extends CustomPainter {
  _BoxPainter({required this.box, required this.image, required this.color, required this.label});

  final Rect box;
  final Size image;
  final Color color;
  final String label;

  @override
  void paint(Canvas canvas, Size size) {
    final sx = size.width / image.width, sy = size.height / image.height;
    final r = Rect.fromLTRB(box.left * sx, box.top * sy, box.right * sx, box.bottom * sy).inflate(2);
    canvas.drawRect(Offset.zero & size, Paint()..color = const Color(0x55000000));
    canvas.save();
    canvas.clipRect(r);
    canvas.drawColor(const Color(0x00000000), BlendMode.clear);
    canvas.restore();
    canvas.drawRRect(
      RRect.fromRectAndRadius(r, const Radius.circular(4)),
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3,
    );
    final text = TextPainter(
      text: TextSpan(
        text: ' $label ',
        style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.bold),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final top = r.top - text.height - 2 >= 0 ? r.top - text.height - 2 : r.bottom + 2;
    final at = Offset(r.left.clamp(0, math.max(0, size.width - text.width)), top);
    canvas.drawRRect(RRect.fromRectAndRadius(at & text.size, const Radius.circular(3)), Paint()..color = color);
    text.paint(canvas, at);
  }

  @override
  bool shouldRepaint(_BoxPainter old) => old.box != box || old.color != color || old.label != label;
}

class _Intro extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('판독관이 볼 사진을 줄입니다', style: t.titleLarge?.copyWith(fontWeight: FontWeight.bold)),
        const SizedBox(height: 6),
        Text('위해물품 12종마다 통과 · 재검 · 적발로 나눕니다. 놓침 5% 상한을 지키도록 정한 기준이며, 재검은 반드시 사람이 봅니다.', style: t.bodyMedium),
      ],
    );
  }
}

class _ZoneGuide extends StatelessWidget {
  const _ZoneGuide();

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('판정 구간', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        for (final zone in JudgmentZone.values)
          Card(
            margin: const EdgeInsets.only(bottom: 8),
            child: ListTile(
              leading: CircleAvatar(backgroundColor: zone.color, radius: 8),
              title: Text(zone.label, style: const TextStyle(fontWeight: FontWeight.bold)),
              subtitle: Text(zone.description),
              trailing: zone.needsReview ? const Icon(Icons.person_outline) : null,
            ),
          ),
      ],
    );
  }
}

class _ErrorCard extends StatelessWidget {
  const _ErrorCard({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Theme.of(context).colorScheme.errorContainer,
      child: ListTile(leading: const Icon(Icons.error_outline), title: Text(message)),
    );
  }
}

class _ResultView extends StatelessWidget {
  const _ResultView({required this.result, required this.selected, required this.onSelect});

  final Inspection result;
  final String? selected;
  final ValueChanged<String> onSelect;

  String _headline(Inspection r) => switch (r.overall) {
    JudgmentZone.autoAlarm =>
      '개봉 검사: ${r.inZone(JudgmentZone.autoAlarm).map((e) => e.label).join(', ')}'
          '${r.inZone(JudgmentZone.review).isEmpty ? '' : '\n함께 확인: ${r.inZone(JudgmentZone.review).map((e) => e.label).join(', ')}'}',
    JudgmentZone.review => '판독관 확인 필요: ${r.inZone(JudgmentZone.review).map((e) => e.label).join(', ')}',
    JudgmentZone.autoClear => '12종 모두 기준 아래 — 판독관 확인 없이 통과',
  };

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    final overall = result.overall;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(color: overall.color, borderRadius: BorderRadius.circular(16)),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('사진 판정', style: t.labelLarge?.copyWith(color: Colors.white70)),
              Text(
                overall.label,
                key: const Key('overall'),
                style: t.displaySmall?.copyWith(color: Colors.white, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              Text(_headline(result), style: t.titleMedium?.copyWith(color: Colors.white)),
            ],
          ),
        ),
        const SizedBox(height: 16),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (final zone in JudgmentZone.values.reversed)
              Chip(
                avatar: CircleAvatar(backgroundColor: zone.color, radius: 6),
                label: Text('${zone.label} ${result.inZone(zone).length}종'),
              ),
          ],
        ),
        const SizedBox(height: 12),
        Text('품목별 판정', style: t.titleMedium),
        Text('품목 이름을 누르면 사진에 그 품목이 가장 의심되는 자리를 표시해요', style: t.bodySmall),
        const SizedBox(height: 4),
        for (final item in result.sorted)
          _ItemRow(
            item: item,
            selected: item.item == selected,
            onTap: item.box == null ? null : () => onSelect(item.item),
          ),
        const SizedBox(height: 12),
        Text(
          '점수 막대는 로그 눈금 · 세로선은 통과/적발 기준 · 모델 ${shortVersion(result.modelVersion)}${result.experimental ? ' · 실험용: PIDray 시험 분포에서 정한 기준이며 실제 공항 장비에서 검증하지 않았습니다' : ''}',
          style: t.bodySmall?.copyWith(color: Theme.of(context).colorScheme.outline),
        ),
      ],
    );
  }
}

/// 점수는 0.00001 부터 1 까지 넓게 퍼져 있어서, 막대는 로그 눈금(10^-6 ~ 1)으로 그린다.
/// 직선 눈금이면 기준선이 모두 왼쪽 끝에 몰려 왜 재검인지 보이지 않는다.
double scorePosition(double v) {
  if (v.isNaN || v <= 0) return 0;
  return ((math.log(v) / math.ln10 + 6) / 6).clamp(0.0, 1.0);
}

String formatScore(double v) => v >= 0.001 ? v.toStringAsFixed(3) : (v <= 0 ? '0' : v.toStringAsExponential(0));

String shortVersion(String v) => v.startsWith('sha256:') && v.length > 19 ? v.substring(0, 19) : v;

class _ItemRow extends StatelessWidget {
  const _ItemRow({required this.item, this.selected = false, this.onTap});

  final ItemJudgment item;
  final bool selected;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    return InkWell(
      key: Key('item-${item.item}'),
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 6),
        decoration: BoxDecoration(
          color: selected ? item.zone.color.withValues(alpha: 0.12) : null,
          border: Border.all(color: selected ? item.zone.color : Colors.transparent, width: 1.5),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          children: [
            SizedBox(
              width: 84,
              child: Text(
                item.label,
                style: t.bodyLarge?.copyWith(
                  fontWeight: FontWeight.w600,
                  decoration: onTap == null ? null : TextDecoration.underline,
                  decorationColor: item.zone.color,
                ),
              ),
            ),
            Expanded(
              child: LayoutBuilder(
                builder: (context, box) {
                  final w = box.maxWidth;
                  double at(double v) => scorePosition(v) * w;
                  return SizedBox(
                    height: 18,
                    child: Stack(
                      children: [
                        Container(
                          decoration: BoxDecoration(
                            color: const Color(0xFFEDEEF0),
                            borderRadius: BorderRadius.circular(9),
                          ),
                        ),
                        Container(
                          width: at(item.score),
                          decoration: BoxDecoration(color: item.zone.color, borderRadius: BorderRadius.circular(9)),
                        ),
                        if (!item.low.isNaN)
                          Positioned(
                            left: at(item.low),
                            top: 0,
                            bottom: 0,
                            child: Container(width: 2, color: Colors.black38),
                          ),
                        if (!item.high.isNaN)
                          Positioned(
                            left: at(item.high) - 2,
                            top: 0,
                            bottom: 0,
                            child: Container(width: 2, color: Colors.black54),
                          ),
                      ],
                    ),
                  );
                },
              ),
            ),
            const SizedBox(width: 10),
            SizedBox(
              width: 60,
              child: Text(formatScore(item.score), textAlign: TextAlign.right, style: t.bodySmall),
            ),
            const SizedBox(width: 10),
            Container(
              width: 48,
              padding: const EdgeInsets.symmetric(vertical: 3),
              decoration: BoxDecoration(color: item.zone.color, borderRadius: BorderRadius.circular(999)),
              child: Text(
                item.zone.label,
                textAlign: TextAlign.center,
                style: t.labelMedium?.copyWith(color: Colors.white),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
