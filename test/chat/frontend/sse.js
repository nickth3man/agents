(function() {
  function enforceLightMode() {
    if (document.body && document.body.classList.contains('dark')) {
      document.body.classList.remove('dark');
    }
  }
  enforceLightMode();
  var _modeObs = new MutationObserver(enforceLightMode);
  _modeObs.observe(document.documentElement, { childList: true, subtree: false });
  document.addEventListener('DOMContentLoaded', function() {
    enforceLightMode();
    _modeObs.observe(document.body, { attributes: true, attributeFilter: ['class'] });
  });

  var source = null;
  var _runComplete = false;

  // Phase → bubble element id mapping
  // Round 0: model_a → bubble-model_a, etc.
  // Round 1: round1_model_a → bubble-model_a_r1, etc.
  // Judge: judge → bubble-judge
  var BUBBLE_MAP = {
    'model_a':       'bubble-model_a',
    'model_b':       'bubble-model_b',
    'model_c':       'bubble-model_c',
    'round1_model_a': 'bubble-model_a_r1',
    'round1_model_b': 'bubble-model_b_r1',
    'round1_model_c': 'bubble-model_c_r1',
    'judge':         'bubble-judge'
  };

  var ALL_BUBBLES = [
    'bubble-model_a', 'bubble-model_b', 'bubble-model_c',
    'bubble-model_a_r1', 'bubble-model_b_r1', 'bubble-model_c_r1',
    'bubble-judge'
  ];

  var ROUND_LABELS = {
    'model_a': 'round-label-0',
    'round1_model_a': 'round-label-1'
  };

  var currentBubble = null;

  function setBubbleState(bubbleId, state) {
    var el = document.getElementById(bubbleId);
    if (!el) return;
    el.setAttribute('data-state', state);
    var status = el.querySelector('.bubble-status .label');
    if (status) {
      status.textContent = state === 'active' ? 'Speaking…' :
                           state === 'done'   ? 'Done'       :
                                                 'Awaiting';
    }
  }

  function setStatus(label, mode) {
    var foot = document.getElementById('floor-foot');
    if (!foot) return;
    foot.setAttribute('data-conn', mode || 'idle');
    var lbl = foot.querySelector('.label');
    if (lbl) lbl.textContent = label;
  }

  function clearAll() {
    ALL_BUBBLES.forEach(function(id) { setBubbleState(id, 'idle'); });
    // Clear stream text
    ALL_BUBBLES.forEach(function(id) {
      var el = document.getElementById(id);
      if (el) {
        var s = el.querySelector('.bubble-stream');
        if (s) s.textContent = '';
      }
    });
    // Hide round labels and judge divider
    document.querySelectorAll('.conv-round-label, .conv-divider').forEach(function(el) {
      el.style.display = 'none';
    });
    currentBubble = null;
  }

  function showRoundLabel(phase) {
    var labelId = ROUND_LABELS[phase];
    if (labelId) {
      var el = document.getElementById(labelId);
      if (el) el.style.display = '';
    }
    // Show judge divider
    if (phase === 'judge') {
      var div = document.getElementById('judge-divider');
      if (div) div.style.display = '';
    }
  }

  function appendToken(token) {
    if (!currentBubble) currentBubble = 'bubble-model_a';
    var el = document.getElementById(currentBubble);
    if (!el) return;
    var s = el.querySelector('.bubble-stream');
    if (!s) return;
    s.textContent += token;
    s.scrollTop = s.scrollHeight;
    // Auto-scroll the bubble into view
    el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function startPhase(phase) {
    if (phase === 'reset') {
      clearAll();
      return;
    }
    if (_runComplete) {
      clearAll();
      _runComplete = false;
    }

    var bubbleId = BUBBLE_MAP[phase];
    if (!bubbleId) return;

    // Mark previous as done
    if (currentBubble && currentBubble !== bubbleId) {
      setBubbleState(currentBubble, 'done');
    }

    showRoundLabel(phase);
    currentBubble = bubbleId;
    setBubbleState(bubbleId, 'active');
  }

  function connect() {
    if (source) return;
    source = new EventSource('http://localhost:7861/stream');
    setStatus('Connecting', 'waiting');

    source.onmessage = function(e) {
      try {
        var d = JSON.parse(e.data);
        if (d.token != null) {
          appendToken(d.token);
          setStatus('Live · streaming', 'live');
        }
      } catch(_) {}
    };
    source.addEventListener('phase', function(e) {
      try {
        var d = JSON.parse(e.data);
        if (d.phase) startPhase(d.phase);
      } catch(_) {}
    });
    source.addEventListener('done', function() {
      if (currentBubble) setBubbleState(currentBubble, 'done');
      currentBubble = null;
      _runComplete = true;
      setStatus('Court adjourned', 'idle');
      source.close(); source = null;
      setTimeout(function() {
        setStatus('Awaiting query', 'idle');
        connect();
      }, 1800);
    });
    source.onerror = function() {
      if (source) { source.close(); source = null; }
      setStatus('Reconnecting', 'waiting');
      setTimeout(connect, 1500);
    };
  }

  var obs = new MutationObserver(function() {
    if (document.getElementById('bubble-model_a') && document.getElementById('floor-foot')) {
      obs.disconnect();
      setStatus('Awaiting query', 'idle');
      connect();
    }
  });
  obs.observe(document.body, { childList: true, subtree: true });
})();

;(function() {
  function watchClear() {
    var btn = document.getElementById('clear-btn');
    if (btn) {
      btn.addEventListener('click', function() {
        // Use the global clearAll from the first IIFE
        // We need to find bubbles and reset them
        document.querySelectorAll('.conv-bubble').forEach(function(el) {
          el.setAttribute('data-state', 'idle');
          var s = el.querySelector('.bubble-stream');
          if (s) s.textContent = '';
          var label = el.querySelector('.bubble-status .label');
          if (label) label.textContent = 'Awaiting';
        });
        document.querySelectorAll('.conv-round-label, .conv-divider').forEach(function(el) {
          el.style.display = 'none';
        });
        var foot = document.getElementById('floor-foot');
        if (foot) {
          foot.setAttribute('data-conn', 'idle');
          var lbl = foot.querySelector('.label');
          if (lbl) lbl.textContent = 'Awaiting query';
        }
      });
      return;
    }
    setTimeout(watchClear, 300);
  }
  watchClear();
})();
