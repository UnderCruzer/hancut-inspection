import 'dart:convert';
import 'dart:js_interop';
import 'dart:typed_data';

import 'api.dart';
import 'judgment.dart';

const browserInspectionAvailable = true;

@JS('hancutWarmup')
external JSPromise<JSString> _warmup();

@JS('hancutInspect')
external JSPromise<JSString> _inspect(JSUint8Array bytes);

String _message(Object e) {
  final text = e.toString();
  final korean = RegExp(r'[가-힣].*').firstMatch(text)?.group(0);
  return korean ?? '이 브라우저에서 판정하지 못했어요.';
}

class BrowserInspector implements Inspector {
  @override
  String get description => '이 브라우저에서 판정 — 사진은 밖으로 나가지 않아요';

  @override
  Future<void> warmup() async {
    try {
      await _warmup().toDart;
    } catch (e) {
      throw InspectionError(_message(e));
    }
  }

  @override
  Future<Inspection> inspect(Uint8List bytes, {String filename = 'xray'}) async {
    if (sniffImageType(bytes) == null) {
      throw const InspectionError('JPEG 또는 PNG 사진만 판정할 수 있어요.');
    }
    final String body;
    try {
      body = (await _inspect(bytes.toJS).toDart).toDart;
    } catch (e) {
      throw InspectionError(_message(e));
    }
    return Inspection.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }
}
