import 'package:flutter/material.dart';

import 'judgment.dart';

/// 앱 골격 — 촬영과 판정 요청은 W9에 붙인다 (docs/plan.md).
/// 지금은 판정 구간이 화면에 어떻게 보이는지만 확인하는 화면이다.
class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('한컷점검')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Text('점검 흐름', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          const Text('촬영 → 판정 → 확인·정정 → 점검표 초안'),
          const SizedBox(height: 24),
          Text('판정 구간', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          for (final zone in JudgmentZone.values) _ZoneTile(zone: zone),
          const SizedBox(height: 24),
          FilledButton.icon(
            onPressed: null, // 촬영 화면은 W9
            icon: const Icon(Icons.photo_camera_outlined),
            label: const Text('점검 시작 (준비 중)'),
          ),
        ],
      ),
    );
  }
}

class _ZoneTile extends StatelessWidget {
  const _ZoneTile({required this.zone});

  final JudgmentZone zone;

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: CircleAvatar(backgroundColor: zone.color, radius: 8),
        title: Text(zone.label, style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text(zone.description),
        trailing: zone.needsReview ? const Icon(Icons.person_outline) : null,
      ),
    );
  }
}
