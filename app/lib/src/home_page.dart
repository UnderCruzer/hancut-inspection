import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'api.dart';
import 'judgment.dart';

/// 판독 화면 — X-ray 사진을 고르면 판정 API 로 보내고, 사진 판정과 12종 품목별 구간을 보여준다.
/// 재검이 하나라도 있으면 판독관이 직접 확인해야 한다.
class HomePage extends StatefulWidget {
  const HomePage({super.key, this.pickImage, this.clientFor});

  /// 시험에서 사진 고르기와 서버를 바꿔 끼우기 위한 자리.
  final Future<(Uint8List, String)?> Function()? pickImage;
  final InspectionClient Function(String baseUrl)? clientFor;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _server = TextEditingController(text: const String.fromEnvironment('HANCUT_API', defaultValue: 'http://localhost:8000'));
  Uint8List? _image;
  String? _name;
  Inspection? _result;
  String? _error;
  bool _busy = false;

  @override
  void dispose() {
    _server.dispose();
    super.dispose();
  }

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
      _error = null;
      _busy = true;
    });
    try {
      final client = widget.clientFor?.call(_server.text.trim()) ?? InspectionClient(_server.text.trim());
      final result = await client.inspect(bytes, filename: name.split('.').first);
      if (mounted) setState(() => _result = result);
    } on InspectionError catch (e) {
      if (mounted) setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final wide = MediaQuery.sizeOf(context).width >= 900;
    final left = <Widget>[
      _Intro(),
      const SizedBox(height: 16),
      TextField(
        controller: _server,
        decoration: const InputDecoration(labelText: '판정 서버 주소', border: OutlineInputBorder(), isDense: true),
      ),
      const SizedBox(height: 12),
      FilledButton.icon(
        onPressed: _busy ? null : _inspect,
        icon: const Icon(Icons.image_search),
        label: Text(_busy ? '판정 중…' : 'X-ray 사진 판정하기'),
      ),
      if (_image != null) ...[
        const SizedBox(height: 16),
        ClipRRect(borderRadius: BorderRadius.circular(12), child: Image.memory(_image!, fit: BoxFit.contain,
                errorBuilder: (_, _, _) => const SizedBox(height: 80, child: Center(child: Text('미리보기를 그릴 수 없는 사진이에요'))))),
        const SizedBox(height: 4),
        Text(_name ?? '', style: Theme.of(context).textTheme.bodySmall),
      ],
    ];
    final right = <Widget>[
      if (_error != null) _ErrorCard(message: _error!),
      if (_result != null) _ResultView(result: _result!),
      if (_result == null && _error == null) const _ZoneGuide(),
    ];
    return Scaffold(
      appBar: AppBar(title: const Text('한컷점검 · X-ray 판독 보조')),
      body: wide
          ? Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              SizedBox(width: 440, child: ListView(padding: const EdgeInsets.all(20), children: left)),
              const VerticalDivider(width: 1),
              Expanded(child: ListView(padding: const EdgeInsets.all(20), children: right)),
            ])
          : ListView(padding: const EdgeInsets.all(16), children: [...left, const SizedBox(height: 20), ...right]),
    );
  }
}

class _Intro extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('판독관이 볼 사진을 줄입니다', style: t.titleLarge?.copyWith(fontWeight: FontWeight.bold)),
      const SizedBox(height: 6),
      Text('위해물품 12종마다 통과 · 재검 · 적발로 나눕니다. 놓침 5% 상한을 지키도록 정한 기준이며, 재검은 반드시 사람이 봅니다.',
          style: t.bodyMedium),
    ]);
  }
}

class _ZoneGuide extends StatelessWidget {
  const _ZoneGuide();

  @override
  Widget build(BuildContext context) {
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
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
    ]);
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
  const _ResultView({required this.result});

  final Inspection result;

  String _headline(Inspection r) => switch (r.overall) {
        JudgmentZone.autoAlarm => '개봉 검사: ${r.inZone(JudgmentZone.autoAlarm).map((e) => e.label).join(', ')}'
            '${r.inZone(JudgmentZone.review).isEmpty ? '' : '\n함께 확인: ${r.inZone(JudgmentZone.review).map((e) => e.label).join(', ')}'}',
        JudgmentZone.review => '판독관 확인 필요: ${r.inZone(JudgmentZone.review).map((e) => e.label).join(', ')}',
        JudgmentZone.autoClear => '12종 모두 기준 아래 — 판독관 확인 없이 통과',
      };

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    final overall = result.overall;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(color: overall.color, borderRadius: BorderRadius.circular(16)),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('사진 판정', style: t.labelLarge?.copyWith(color: Colors.white70)),
          Text(overall.label, key: const Key('overall'), style: t.displaySmall?.copyWith(color: Colors.white, fontWeight: FontWeight.bold)),
          const SizedBox(height: 4),
          Text(_headline(result), style: t.titleMedium?.copyWith(color: Colors.white)),
        ]),
      ),
      const SizedBox(height: 16),
      Wrap(spacing: 8, runSpacing: 8, children: [
        for (final zone in JudgmentZone.values.reversed)
          Chip(
            avatar: CircleAvatar(backgroundColor: zone.color, radius: 6),
            label: Text('${zone.label} ${result.inZone(zone).length}종'),
          ),
      ]),
      const SizedBox(height: 12),
      Text('품목별 판정', style: t.titleMedium),
      const SizedBox(height: 4),
      for (final item in result.sorted) _ItemRow(item: item),
      const SizedBox(height: 12),
      Text(
        '점수 막대는 로그 눈금 · 세로선은 통과/적발 기준 · 모델 ${shortVersion(result.modelVersion)}${result.experimental ? ' · 실험용: PIDray 시험 분포에서 정한 기준이며 실제 공항 장비에서 검증하지 않았습니다' : ''}',
        style: t.bodySmall?.copyWith(color: Theme.of(context).colorScheme.outline),
      ),
    ]);
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
  const _ItemRow({required this.item});

  final ItemJudgment item;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(children: [
        SizedBox(width: 84, child: Text(item.label, style: t.bodyLarge?.copyWith(fontWeight: FontWeight.w600))),
        Expanded(
          child: LayoutBuilder(builder: (context, box) {
            final w = box.maxWidth;
            double at(double v) => scorePosition(v) * w;
            return SizedBox(
              height: 18,
              child: Stack(children: [
                Container(decoration: BoxDecoration(color: const Color(0xFFEDEEF0), borderRadius: BorderRadius.circular(9))),
                Container(width: at(item.score), decoration: BoxDecoration(color: item.zone.color, borderRadius: BorderRadius.circular(9))),
                if (!item.low.isNaN) Positioned(left: at(item.low), top: 0, bottom: 0, child: Container(width: 2, color: Colors.black38)),
                if (!item.high.isNaN) Positioned(left: at(item.high) - 2, top: 0, bottom: 0, child: Container(width: 2, color: Colors.black54)),
              ]),
            );
          }),
        ),
        const SizedBox(width: 10),
        SizedBox(width: 60, child: Text(formatScore(item.score), textAlign: TextAlign.right, style: t.bodySmall)),
        const SizedBox(width: 10),
        Container(
          width: 48,
          padding: const EdgeInsets.symmetric(vertical: 3),
          decoration: BoxDecoration(color: item.zone.color, borderRadius: BorderRadius.circular(999)),
          child: Text(item.zone.label, textAlign: TextAlign.center, style: t.labelMedium?.copyWith(color: Colors.white)),
        ),
      ]),
    );
  }
}
