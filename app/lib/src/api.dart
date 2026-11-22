import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

import 'judgment.dart';

/// 판정 API 호출이 실패한 이유. 화면에 그대로 보여줄 한국어 문장을 담는다.
class InspectionError implements Exception {
  const InspectionError(this.message);

  final String message;

  @override
  String toString() => message;
}

/// 사진 내용으로 형식을 정한다. PIDray hidden 은 .png 이름의 JPEG 이라 확장자를 믿지 않는다.
String? sniffImageType(Uint8List bytes) {
  if (bytes.length >= 8 && bytes[0] == 0x89 && bytes[1] == 0x50 && bytes[2] == 0x4E && bytes[3] == 0x47) {
    return 'png';
  }
  if (bytes.length >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF) {
    return 'jpeg';
  }
  return null;
}

class InspectionClient {
  InspectionClient(this.baseUrl, {http.Client? client}) : _client = client ?? http.Client();

  final String baseUrl;
  final http.Client _client;

  Uri _uri(String path) => Uri.parse(baseUrl.replaceAll(RegExp(r'/+$'), '') + path);

  Future<Inspection> inspect(Uint8List bytes, {String filename = 'xray'}) async {
    final type = sniffImageType(bytes);
    if (type == null) {
      throw const InspectionError('JPEG 또는 PNG 사진만 판정할 수 있어요.');
    }
    final request = http.MultipartRequest('POST', _uri('/v2/inspections'))
      ..files.add(http.MultipartFile.fromBytes('image', bytes,
          filename: '$filename.${type == 'png' ? 'png' : 'jpg'}', contentType: MediaType('image', type)));
    final http.Response response;
    try {
      response = await http.Response.fromStream(await _client.send(request)).timeout(const Duration(seconds: 60));
    } catch (_) {
      throw InspectionError('판정 서버에 연결하지 못했어요. 주소($baseUrl)와 서버 실행을 확인해 주세요.');
    }
    final body = utf8.decode(response.bodyBytes);
    if (response.statusCode != 200) {
      throw InspectionError(switch (response.statusCode) {
        503 => '판정 모델이 아직 준비되지 않았어요 (서버 503).',
        413 => '사진이 너무 커요.',
        415 || 400 => '읽을 수 없는 사진이에요.',
        _ => '판정에 실패했어요 (서버 ${response.statusCode}).',
      });
    }
    try {
      return Inspection.fromJson(jsonDecode(body) as Map<String, dynamic>);
    } catch (_) {
      throw const InspectionError('서버 응답을 읽지 못했어요.');
    }
  }
}
