import { useState } from 'react';
import { X } from 'lucide-react';

export interface UserProfile {
	name: string;
	phone: string;
	email: string;
	address: string;
	city: string;
	purpose_of_visit: string;
	medical_history?: string;
	allergies?: string;
	notes?: string;
}

interface EnquiryFormProps {
	isOpen: boolean;
	onClose: () => void;
	onSubmit: (data: UserProfile) => void;
	isLoading?: boolean;
}

const initialForm: UserProfile = {
	name: '', phone: '', email: '', address: '', city: '', purpose_of_visit: '',
	medical_history: '', allergies: '', notes: '',
};

export default function EnquiryForm({ isOpen, onClose, onSubmit, isLoading }: EnquiryFormProps) {
	const [formData, setFormData] = useState<UserProfile>(initialForm);
	if (!isOpen) return null;

	const update = (name: string, value: string) => setFormData((current) => ({ ...current, [name]: value }));
	const submit = (event: React.FormEvent) => {
		event.preventDefault();
		const required = ['name', 'phone', 'email', 'address', 'city', 'purpose_of_visit'] as const;
		if (required.some((key) => !formData[key])) {
			alert('Please fill all required fields');
			return;
		}
		onSubmit(formData);
	};

	return <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
		<div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-xl bg-white text-slate-900">
			<header className="flex items-center justify-between bg-cyan-600 p-5 text-white"><h2 className="text-xl font-semibold">Patient information</h2><button aria-label="Close" onClick={onClose}><X /></button></header>
			<form onSubmit={submit} className="grid gap-4 p-6 sm:grid-cols-2">
				{(['name', 'phone', 'email', 'city', 'address'] as const).map((field) => <input key={field} required={field !== 'email'} className="rounded-lg border border-slate-300 px-3 py-2" placeholder={field.replace('_', ' ')} value={formData[field]} onChange={(event) => update(field, event.target.value)} />)}
				<select required className="rounded-lg border border-slate-300 px-3 py-2 sm:col-span-2" value={formData.purpose_of_visit} onChange={(event) => update('purpose_of_visit', event.target.value)}><option value="">Purpose of visit</option><option>Routine Checkup</option><option>Cleaning</option><option>Root Canal</option><option>Braces</option><option>Emergency Treatment</option><option>Other</option></select>
				{(['medical_history', 'allergies', 'notes'] as const).map((field) => <textarea key={field} className="rounded-lg border border-slate-300 px-3 py-2 sm:col-span-2" rows={2} placeholder={`${field.replace('_', ' ')} (optional)`} value={formData[field]} onChange={(event) => update(field, event.target.value)} />)}
				<button type="button" onClick={onClose} className="rounded-lg border border-slate-300 px-4 py-3">Cancel</button><button type="submit" disabled={isLoading} className="rounded-lg bg-cyan-600 px-4 py-3 font-semibold text-white disabled:opacity-50">{isLoading ? 'Starting...' : 'Start voice call'}</button>
			</form>
		</div>
	</div>;
}
