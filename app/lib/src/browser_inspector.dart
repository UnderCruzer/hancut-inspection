/// 브라우저 안 판정 (#62). 웹에서는 `web/hancut_infer.js` 를 부르고, 그 밖에서는 쓸 수 없다.
library;

export 'browser_inspector_stub.dart' if (dart.library.js_interop) 'browser_inspector_web.dart';
