// ESLint 9 flat config for the Oelala frontend (Vite 7 + React 18, JSX only).
//
// Purpose: catch undefined-variable bugs statically. `vite build` does not check
// identifiers, so a ReferenceError like "resolved is not defined" only surfaced
// at runtime. `no-undef` is therefore an error here.
//
// Run: npm run lint
//
// Deliberately NOT enabled yet (kept as warn/off to avoid drowning in
// pre-existing findings): formatting/stylistic rules, react/prop-types,
// react-hooks/exhaustive-deps. See changelog/20261001-frontend-eslint.md.

import js from '@eslint/js'
import react from 'eslint-plugin-react'
import reactHooks from 'eslint-plugin-react-hooks'
import globals from 'globals'

export default [
  {
    ignores: [
      'dist/**',
      'node_modules/**',
      'coverage/**',
    ],
  },
  js.configs.recommended,
  {
    files: ['**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
    settings: {
      react: {
        version: 'detect',
        jsxRuntime: 'automatic', // @vitejs/plugin-react uses the automatic JSX runtime
      },
    },
    plugins: {
      react,
      'react-hooks': reactHooks,
    },
    rules: {
      // --- Core ----------------------------------------------------------------
      'no-undef': 'error',
      'no-unused-vars': ['error', {
        argsIgnorePattern: '^_',
        varsIgnorePattern: '^_',
      }],
      // Empty catch blocks are an accepted pattern here (localStorage/sessionStorage
      // calls guarded with try/catch).
      'no-empty': ['error', { allowEmptyCatch: true }],

      // --- React plugin: correctness rules ----------------------------------
      // jsx-uses-vars: a JSX-referenced component counts as a variable use,
      // so `no-unused-vars` must not flag it.
      'react/jsx-uses-vars': 'error',
      'react/jsx-key': 'error',
      'react/jsx-no-duplicate-props': 'error',
      'react/jsx-no-undef': 'error',
      'react/jsx-no-target-blank': 'error',
      'react/jsx-no-useless-fragment': 'warn',
      'react/no-children-prop': 'error',
      'react/no-direct-mutation-state': 'error',
      'react/no-unknown-property': 'warn',
      'react/no-unescaped-entities': 'warn',
      'react/no-string-refs': 'error',
      'react/no-find-dom-node': 'error',
      'react/no-is-mounted': 'error',
      'react/no-render-return-value': 'error',
      'react/no-danger-with-children': 'error',
      'react/display-name': 'error', // measured 0 findings today — free future protection

      // --- React plugin: deliberately off (noise, not defects) ---------------
      // prop-types: measured 525 findings in 53 files (codebase uses no PropTypes).
      'react/prop-types': 'off',
      // react-in-jsx-scope: N/A — Vite uses the automatic JSX runtime.
      'react/react-in-jsx-scope': 'off',

      // --- React hooks --------------------------------------------------------
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
    },
  },
]
