import { AUTHOR, COPYRIGHT_YEAR, GITHUB_URL, HF_URL } from '../site'

const links = [
  { href: '/download', label: 'Download' },
  { href: '/docs', label: 'Docs' },
  { href: GITHUB_URL, label: 'GitHub' },
  { href: HF_URL, label: 'Hugging Face' },
  { href: '/search', label: 'Models' },
  { href: '/results', label: 'Results' },
]

export function Footer() {
  return (
    <footer class="mt-24 border-t border-line">
      <div class="mx-auto flex max-w-6xl flex-col-reverse gap-4 px-4 py-8 text-xs text-muted sm:flex-row sm:items-center sm:justify-between md:px-6">
        <p>
          © {COPYRIGHT_YEAR} Ollaya. Built by{' '}
          <a href={AUTHOR.url} class="text-body underline-offset-4 hover:text-fg hover:underline">
            {AUTHOR.name}
          </a>
          .
        </p>
        <nav aria-label="Footer">
          <ul class="flex flex-wrap gap-x-5 gap-y-2">
            {links.map((l) => (
              <li>
                <a href={l.href} class="underline-offset-4 hover:text-fg hover:underline">
                  {l.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </footer>
  )
}
