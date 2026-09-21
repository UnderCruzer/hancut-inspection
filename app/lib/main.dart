import 'package:flutter/material.dart';

import 'src/home_page.dart';

void main() {
  runApp(const HancutApp());
}

class HancutApp extends StatelessWidget {
  const HancutApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '한컷점검',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFD9432A)),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}
