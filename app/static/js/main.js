// Flask RBAC Enterprise Frontend Utilities

document.addEventListener('DOMContentLoaded', () => {
  // 1. Auto dismiss alerts after 5 seconds
  setTimeout(() => {
    document.querySelectorAll('.alert').forEach(el => {
      el.style.transition = 'opacity 0.4s ease';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 400);
    });
  }, 5000);

  // 2. Modals handling
  window.openModal = function(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.add('active');
  };

  window.closeModal = function(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.remove('active');
  };

  // Close modal when clicking backdrop
  document.querySelectorAll('.modal-overlay').forEach(overlay => {
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        overlay.classList.remove('active');
      }
    });
  });

  // 3. User table search filter
  const userSearchInput = document.getElementById('tableSearchInput');
  if (userSearchInput) {
    userSearchInput.addEventListener('input', (e) => {
      const term = e.target.value.toLowerCase();
      document.querySelectorAll('.table-row-item').forEach(row => {
        const text = row.innerText.toLowerCase();
        row.style.display = text.includes(term) ? '' : 'none';
      });
    });
  }

  // 4. Interactive API Tester
  window.executeApiCall = async function(endpoint, method = 'GET', body = null, resultContainerId) {
    const container = document.getElementById(resultContainerId);
    if (!container) return;

    container.innerHTML = '<span style="color: #94a3b8;">Executing request...</span>';

    try {
      const options = {
        method: method,
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      };

      if (body && (method === 'POST' || method === 'PUT')) {
        options.body = typeof body === 'string' ? body : JSON.stringify(body);
      }

      const startTime = performance.now();
      const res = await fetch(endpoint, options);
      const elapsed = Math.round(performance.now() - startTime);

      const contentType = res.headers.get('content-type');
      let data;
      if (contentType && contentType.includes('application/json')) {
        data = await res.json();
      } else {
        data = await res.text();
      }

      const statusBadge = res.ok 
        ? `<span class="badge badge-success">${res.status} ${res.statusText}</span>`
        : `<span class="badge badge-danger">${res.status} ${res.statusText}</span>`;

      container.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <div>Status: ${statusBadge} <span style="font-size: 0.8rem; color: #94a3b8; margin-left: 8px;">Time: ${elapsed}ms</span></div>
          <button class="btn btn-sm btn-secondary" onclick="navigator.clipboard.writeText('${JSON.stringify(data).replace(/'/g, "\\'")}')">Copy JSON</button>
        </div>
        <pre class="code-response"><code>${JSON.stringify(data, null, 2)}</code></pre>
      `;
    } catch (err) {
      container.innerHTML = `<span style="color: #f43f5e;">Request Failed: ${err.message}</span>`;
    }
  };

  // 5. Toggle API accordion
  window.toggleApiAccordion = function(id) {
    const el = document.getElementById(id);
    if (el) {
      el.classList.toggle('open');
    }
  };
});
