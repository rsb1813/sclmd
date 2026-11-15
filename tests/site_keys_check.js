// node tests/site_keys_check.js deploy/site/assets: the pages' scripts outside a browser (scripts/site_check.py runs
// it). keys.js and site.js run in a bare context (no document: they only define their helpers), then: base64url both
// ways, the WebAuthn JSON forms by hand and through a native parser and toJSON(), every error text, the callback's
// fragment, and site.js's number forms. Prints "ok ..." / "FAIL ..." lines, "FALLBACK <json>", and exits 1 on a FAIL.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const dir = process.argv[2] || path.join(__dirname, "..", "deploy", "site", "assets");
let fails = 0;
function ok(cond, what) {
  if (cond) console.log("ok " + what);
  else { fails++; console.log("FAIL " + what); }
}
function eqBytes(a, b) {
  if (!(a instanceof Uint8Array) || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function load(extra) {
  const window = {};
  const ctx = vm.createContext(Object.assign({ window, atob, btoa, Uint8Array, ArrayBuffer, JSON, Math, String, Number, Array,
    Object, TypeError, Promise, isFinite, isNaN, decodeURIComponent, navigator: {} }, extra || {}));
  for (const f of ["site.js", "keys.js"]) vm.runInContext(fs.readFileSync(path.join(dir, f), "utf8"), ctx, { filename: f });
  return { K: window.SclmdKeys, S: window.SclmdSite, ctx };
}

const { K, S } = load();
ok(K && S, "both scripts load without a document");

// base64url
ok(K.bytesToB64u(new Uint8Array([])) === "", "b64u of nothing");
ok(K.bytesToB64u(new Uint8Array([0xfb, 0xff])) === "-_8", "b64u uses - and _ and no padding");
ok(eqBytes(K.b64uToBytes("-_8"), new Uint8Array([0xfb, 0xff])), "b64u back");
ok(eqBytes(K.b64uToBytes("-_8="), new Uint8Array([0xfb, 0xff])), "b64u with padding is read too");
let round = true;
for (let n = 0; n < 300; n++) {
  const b = new Uint8Array(n);
  for (let i = 0; i < n; i++) b[i] = (i * 37 + n * 11) & 255;
  const s = K.bytesToB64u(b);
  if (/[+/=]/.test(s) || !eqBytes(K.b64uToBytes(s), b)) round = false;
  if (!eqBytes(K.b64uToBytes(K.bytesToB64u(b.buffer)), b)) round = false;
}
ok(round, "b64u round trip, 0..299 bytes, from Uint8Array and ArrayBuffer");
const big = new Uint8Array(200000).map((_, i) => i & 255);
ok(eqBytes(K.b64uToBytes(K.bytesToB64u(big)), big), "b64u of 200,000 bytes (chunked)");
for (const bad of ["a+b", "a/b", "a b", "é", 12, null]) {
  let threw = false;
  try { K.b64uToBytes(bad); } catch (e) { threw = e instanceof TypeError || e.name === "TypeError"; }
  ok(threw, "b64u refuses " + JSON.stringify(bad));
}

// creation and request options by hand (no PublicKeyCredential here)
const created = {
  rp: { id: "ssalmuck-ai.duckdns.org", name: "ssalmuck-ai" },
  user: { id: K.bytesToB64u(new Uint8Array([1, 2, 3, 4])), name: "n", displayName: "d" },
  challenge: K.bytesToB64u(new Uint8Array([9, 8, 7])),
  pubKeyCredParams: [{ type: "public-key", alg: -7 }, { type: "public-key", alg: -257 }],
  timeout: 300000, attestation: "none",
  excludeCredentials: [{ type: "public-key", id: K.bytesToB64u(new Uint8Array([5, 5])), transports: ["internal"] }],
  authenticatorSelection: { residentKey: "required", requireResidentKey: true, userVerification: "preferred" }
};
for (const wrapped of [created, { publicKey: created }]) {
  const p = K.creationOptions(wrapped);
  ok(eqBytes(p.challenge, new Uint8Array([9, 8, 7])) && eqBytes(p.user.id, new Uint8Array([1, 2, 3, 4])) &&
     eqBytes(p.excludeCredentials[0].id, new Uint8Array([5, 5])) && p.excludeCredentials[0].transports[0] === "internal" &&
     p.rp.id === "ssalmuck-ai.duckdns.org" && p.authenticatorSelection.residentKey === "required" && p.user.name === "n",
     "creation options by hand" + (wrapped.publicKey ? " ({publicKey: ...})" : ""));
}
ok(typeof created.challenge === "string", "the server's options are not changed in place");
const req = { challenge: K.bytesToB64u(new Uint8Array([3, 1, 4])), rpId: "ssalmuck-ai.duckdns.org", userVerification: "preferred",
  allowCredentials: [{ type: "public-key", id: K.bytesToB64u(new Uint8Array([2, 7])) }], timeout: 300000 };
const rp = K.requestOptions(req);
ok(eqBytes(rp.challenge, new Uint8Array([3, 1, 4])) && eqBytes(rp.allowCredentials[0].id, new Uint8Array([2, 7])) &&
   rp.rpId === "ssalmuck-ai.duckdns.org", "request options by hand");
ok(eqBytes(K.requestOptions({ challenge: "AQI", rpId: "x" }).challenge, new Uint8Array([1, 2])) &&
   K.requestOptions({ challenge: "AQI" }).allowCredentials === undefined, "discoverable login: no allowCredentials");
let threw = false;
try { K.creationOptions({ challenge: "AQI", user: {} }); } catch (e) { threw = true; }
ok(threw, "creation options without a user id are refused");

// a native parser, when the browser has one; a parser that throws falls back to the hand-made form
{
  const native = { parseCreationOptionsFromJSON: (o) => ({ native: "create", o }), parseRequestOptionsFromJSON: (o) => ({ native: "get", o }) };
  const { K: KN } = load({ PublicKeyCredential: native });
  ok(KN.creationOptions({ publicKey: created }).native === "create" && KN.creationOptions(created).o === created, "native creation parser used");
  ok(KN.requestOptions(req).native === "get", "native request parser used");
  const broken = { parseCreationOptionsFromJSON: () => { throw new TypeError("x"); }, parseRequestOptionsFromJSON: () => { throw new TypeError("x"); } };
  const { K: KB } = load({ PublicKeyCredential: broken });
  ok(eqBytes(KB.creationOptions(created).challenge, new Uint8Array([9, 8, 7])), "a native parser that throws: by hand");
  ok(eqBytes(KB.requestOptions(req).challenge, new Uint8Array([3, 1, 4])), "a native request parser that throws: by hand");
}

// credentials to JSON
const buf = (a) => new Uint8Array(a).buffer;
const attestation = {
  id: K.bytesToB64u(new Uint8Array([7, 7, 7])), rawId: buf([7, 7, 7]), type: "public-key", authenticatorAttachment: "platform",
  getClientExtensionResults: () => ({ credProps: { rk: true } }),
  response: { clientDataJSON: buf([123, 125]), attestationObject: buf([1, 2, 3]), getTransports: () => ["internal", "hybrid"],
    getPublicKeyAlgorithm: () => -7, getPublicKey: () => buf([4, 4]), getAuthenticatorData: () => buf([6]) }
};
let j = K.credentialJSON(attestation);
ok(j.id === "BwcH" && j.rawId === "BwcH" && j.type === "public-key" && j.response.clientDataJSON === "e30" &&
   j.response.attestationObject === "AQID" && j.response.transports.join() === "internal,hybrid" && j.response.publicKeyAlgorithm === -7 &&
   j.response.publicKey === "BAQ" && j.response.authenticatorData === "Bg" && j.authenticatorAttachment === "platform" &&
   j.clientExtensionResults.credProps.rk === true, "registration JSON by hand");
const sparse = { id: "AQ", rawId: buf([1]), type: "public-key", response: { clientDataJSON: buf([1]), attestationObject: buf([2]),
  getPublicKey: () => { throw new Error("alg"); } } };
j = K.credentialJSON(sparse);
ok(j.response.transports.length === 0 && !("publicKey" in j.response) && !("authenticatorAttachment" in j) &&
   JSON.stringify(j.clientExtensionResults) === "{}", "registration JSON of an older browser (no getTransports, getPublicKey throws)");
const assertion = { id: "CQk", rawId: buf([9, 9]), type: "public-key",
  response: { clientDataJSON: buf([1]), authenticatorData: buf([2]), signature: buf([3]), userHandle: buf([4]) } };
j = K.credentialJSON(assertion);
ok(j.response.authenticatorData === "Ag" && j.response.signature === "Aw" && j.response.userHandle === "BA" &&
   !("attestationObject" in j.response), "login JSON by hand");
assertion.response.userHandle = null;
ok(K.credentialJSON(assertion).response.userHandle === null, "login JSON without a user handle");
ok(K.credentialJSON(Object.assign({}, assertion, { toJSON: () => ({ id: "native", response: {} }) })).id === "native", "toJSON() used when present");
ok(K.credentialJSON(Object.assign({}, assertion, { toJSON: () => { throw new Error("pm"); } })).id === "CQk", "toJSON() that throws: by hand");

// error texts: every code, status, passkey error; nothing from outside reaches the page but a checked code
const hangul = /[가-힣]/;
let all = true;
for (const code of Object.keys(K.CODES)) {
  const m = K.messageFor({ kind: "api", status: 400, code });
  if (!hangul.test(m.text) || m.detail.indexOf(code) < 0) { all = false; console.log("  code " + code); }
}
ok(all, Object.keys(K.CODES).length + " error codes have a Korean text");
all = true;
for (const st of [400, 401, 403, 404, 405, 413, 415, 429, 500, 501, 502, 503, 504, 599, 418]) {
  const m = K.messageFor({ kind: "api", status: st, code: "" });
  if (!hangul.test(m.text) || m.text === K.CODES.unknown && st >= 500) { all = false; console.log("  status " + st); }
}
ok(all, "every HTTP status has a Korean text");
for (const n of Object.keys(K.WEBAUTHN).concat(["DataError", "EncodingError", "Whatever"])) {
  const m = K.messageFor({ kind: "webauthn", name: n });
  ok(hangul.test(m.text), "passkey error " + n);
}
ok(/분 뒤/.test(K.messageFor({ kind: "api", status: 429, code: "rate_limited", retryAfter: "600" }).text), "429 says when (minutes)");
ok(/시간 뒤/.test(K.messageFor({ kind: "api", status: 429, code: "", retryAfter: "86430" }).text), "429 says when (hours)");
ok(/초 뒤/.test(K.messageFor({ kind: "api", status: 429, code: "too_many_requests", retryAfter: "30" }).text), "429 says when (seconds)");
ok(!/뒤에/.test(K.messageFor({ kind: "api", status: 429, code: "", retryAfter: "soon" }).text), "a Retry-After that is not a number is left out");
ok(/하루에 받을 수 있는 키\(3개\)/.test(K.messageFor({ kind: "api", status: 429, code: "too_many_keys" }).text) &&
   /시간이 지나면/.test(K.messageFor({ kind: "api", status: 429, code: "too_many_keys" }).text), "keys per IP per day, without Retry-After");
for (const c of ["denied", "expired", "browser", "google", "token", "rate_limited", "busy", "bad_request", "server", "bad_body", "invalid_credential",
  "no_session", "csrf", "google_check_needed", "no_key", "unknown_path", "method_not_allowed", "rate_limit_exceeded", "server_error"]) {
  ok(K.CODES[c] && K.CODES[c] !== K.CODES.unknown, "P2's code " + c + " has its own text");
}
ok(hangul.test(K.messageFor({ kind: "network" }).text) && hangul.test(K.messageFor({ kind: "unsupported" }).text) &&
   hangul.test(K.messageFor(null).text), "network, unsupported, nothing");
const evil = K.messageFor({ kind: "api", status: 400, code: "<img src=x onerror=1>" });
ok(evil.detail === "HTTP 400" && evil.text === K.STATUS[400], "a code that is not [a-z0-9_.-] is not shown");
ok(K.messageFor({ kind: "api", status: 0, code: "access_denied" }).detail === "오류 코드 access_denied", "a callback code has no HTTP status");

// the callback's fragment
ok(K.callbackState("#verified").verified === true, "#verified");
ok(K.callbackState("#error=access_denied").error === "access_denied", "#error=access_denied");
ok(K.callbackState("#error=%3Cscript%3E").error === "unknown", "#error=<script> -> unknown");
ok(K.callbackState("#error=%E0%A4%A").error === "unknown", "#error= with broken escapes -> unknown");
ok(K.callbackState("#error=").error === "unknown", "#error= empty -> unknown");
ok(K.callbackState("#error=" + "a".repeat(49)).error === "unknown", "a code longer than 48 -> unknown");
ok(K.callbackState("#something") === null && K.callbackState("") === null, "another fragment is not a callback");

// site.js
ok(S.tokens(1250000) === "125만" && S.tokens(40000000) === "4,000만" && S.tokens(7500000) === "750만" && S.tokens(300000) === "30만" &&
   S.tokens(15000000) === "1,500만" && S.tokens(1234) === "1,234" && S.tokens(-1) === "?", "token numbers in 만");
ok(S.duration(59) === "59초" && S.duration(3725) === "1시간 2분" && S.duration(90061) === "1일 1시간", "durations");
console.log("FALLBACK " + JSON.stringify(S.FALLBACK.limits));

console.log(fails ? fails + " FAILED" : "all ok");
process.exit(fails ? 1 : 0);
