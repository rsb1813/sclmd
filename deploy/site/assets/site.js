/* ssalmuck-ai: the pages' shared script (design.md 6.23). GET /v1/status fills the live state and the limits
   table; buttons with data-copy copy a code block. No storage, no other origin, no inline code (the CSP forbids it). */
(function () {
  "use strict";

  // 6.20's numbers: shown when GET /v1/status cannot be read
  var FALLBACK = {
    limits: {
      ip_unverified: { tph: 300000, tpd: 7500000 },
      ip_verified: { tph: 750000, tpd: 15000000 },
      key: { tph: 1250000, tpd: 40000000 },
      output_weight: 5,
      cached_weight: 0.1,
      concurrency: { priority: 1, normal: 3, low: 8 }
    }
  };
  var TIERS = { ip_unverified: "미검증 IP", ip_verified: "검증 IP", key: "키", local: "운영자 PC" };

  function tokens(n) {
    // 1,250,000 -> "125만", 40,000,000 -> "4,000만", 7,500,000 -> "750만"
    if (typeof n !== "number" || !isFinite(n) || n < 0) return "?";
    if (n >= 10000 && n % 10000 === 0) return (n / 10000).toLocaleString("ko-KR") + "만";
    return n.toLocaleString("ko-KR");
  }

  function duration(s) {
    if (typeof s !== "number" || !isFinite(s) || s < 0) return "?";
    s = Math.floor(s);
    var d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
    if (d) return d + "일 " + h + "시간";
    if (h) return h + "시간 " + m + "분";
    if (m) return m + "분";
    return s + "초";
  }

  function pick(obj, path) {
    var parts = path.split(".");
    for (var i = 0; i < parts.length; i++) {
      if (obj === null || typeof obj !== "object" || !(parts[i] in obj)) return undefined;
      obj = obj[parts[i]];
    }
    return obj;
  }

  function setText(el, text) { if (el) el.textContent = text; }

  function fillLimits(limits, live) {
    var cells = document.querySelectorAll("[data-limit]");
    for (var i = 0; i < cells.length; i++) {
      var v = pick(limits, cells[i].getAttribute("data-limit"));
      if (typeof v === "number") {
        var kind = cells[i].getAttribute("data-format");
        cells[i].textContent = kind === "plain" ? v.toLocaleString("ko-KR") : tokens(v);
      }
    }
    var c = limits && limits.concurrency, refused = document.querySelectorAll("[data-limit-refused]");
    if (c && typeof c.priority === "number" && typeof c.normal === "number" && typeof c.low === "number") {
      for (var k = 0; k < refused.length; k++) refused[k].textContent = String(c.priority + c.normal + c.low + 1);
    }
    var src = document.querySelectorAll("[data-limits-source]");
    for (var j = 0; j < src.length; j++) {
      src[j].textContent = live ? "서버가 알려 준 현재 값입니다." : "서버에 연결하지 못해 공개한 기본값을 표시합니다.";
    }
  }

  function fillCaller(tier) {
    var rows = document.querySelectorAll("tr[data-tier]");
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].getAttribute("data-tier") === tier) rows[i].classList.add("you");
    }
    var els = document.querySelectorAll("[data-caller]");
    for (var j = 0; j < els.length; j++) {
      els[j].textContent = TIERS[tier] ? "이 브라우저의 등급: " + TIERS[tier] : "";
      els[j].hidden = !TIERS[tier];
    }
  }

  function fillState(st) {
    var up = !!(st && st.ok === true);
    var fields = document.querySelectorAll("[data-state]");
    for (var j = 0; j < fields.length; j++) {
      var f = fields[j].getAttribute("data-state"), text = "?";
      if (!st) {
        text = f === "ok" ? "응답 없음" : "?";
      } else if (f === "ok") {
        text = up ? "정상" : "준비되지 않음";
      } else if (f === "uptime_s") {
        text = duration(st.uptime_s);
      } else if (f === "queue") {
        var q = st.queue || {};
        text = "처리 중 " + (q.under_way | 0) + " · 대기 " + (q.waiting | 0);
      } else if (f.indexOf("zdr.") === 0) {
        var z = pick(st, f);
        text = z === false ? "아니요" : z === true ? "예" : (z === undefined || z === "" ? "?" : z === null ? "없음" : String(z));
        if (f === "zdr.cache_ram_minutes" || f === "zdr.cache_disk_hours") {
          text = !z ? "꺼짐" : typeof z === "number" ? z + (f === "zdr.cache_ram_minutes" ? "분" : "시간") : text;
        }
      } else {
        var v = pick(st, f);
        text = v === undefined || v === null || v === "" ? "?" : String(v);
      }
      fields[j].textContent = text;
    }
    var when = document.querySelectorAll("[data-state-time]");
    for (var k = 0; k < when.length; k++) {
      when[k].textContent = new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit" }) + " 기준";
    }
  }

  function loadStatus() {
    var wants = document.querySelector("[data-limit], [data-state], [data-caller]");
    if (!wants || typeof fetch !== "function") return;
    var ctl = typeof AbortController === "function" ? new AbortController() : null;
    var timer = ctl ? setTimeout(function () { ctl.abort(); }, 10000) : null;
    fetch("/v1/status", { method: "GET", credentials: "omit", cache: "no-store", signal: ctl ? ctl.signal : undefined })
      .then(function (r) { if (!r.ok) throw new Error("status " + r.status); return r.json(); })
      .then(function (st) {
        if (timer) clearTimeout(timer);
        var limits = st && st.limits && typeof st.limits === "object" ? st.limits : null;
        fillLimits(limits || FALLBACK.limits, !!limits);
        fillCaller(st && st.caller ? st.caller.tier : null);
        fillState(st);
      })
      .catch(function () {
        if (timer) clearTimeout(timer);
        fillLimits(FALLBACK.limits, false);
        fillState(null);
      });
  }

  function copyText(text, button) {
    function done(ok) {
      var old = button.getAttribute("data-label") || button.textContent;
      button.setAttribute("data-label", old);
      button.textContent = ok ? "복사했습니다" : "복사하지 못했습니다. 직접 선택해 복사하세요";
      setTimeout(function () { button.textContent = old; }, 2500);
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(fallbackCopy(text)); });
    } else {
      done(fallbackCopy(text));
    }
  }

  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.className = "offscreen";
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  }

  function wireCopy() {
    var buttons = document.querySelectorAll("button[data-copy]");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].hidden = false;
      buttons[i].addEventListener("click", function (ev) {
        var target = document.getElementById(ev.currentTarget.getAttribute("data-copy"));
        if (target) copyText(target.textContent, ev.currentTarget);
      });
    }
  }

  window.SclmdSite = { tokens: tokens, duration: duration, copyText: copyText, FALLBACK: FALLBACK, TIERS: TIERS };

  if (typeof document !== "undefined" && document.addEventListener) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () { wireCopy(); loadStatus(); });
    } else {
      wireCopy();
      loadStatus();
    }
  }
})();
